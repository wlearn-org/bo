/*
 * bo.c -- Bayesian Optimization with Gaussian Processes (C11)
 *
 * Full BO engine: search space, GP, kernels, L-BFGS, acquisition,
 * Thompson sampling for categoricals, per-context GPs.
 */

#include "bo.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

/* ========== Error handling ========== */

static _Thread_local char bo_error_buf[256] = {0};

static void bo_set_error(const char *msg) {
    strncpy(bo_error_buf, msg, sizeof(bo_error_buf) - 1);
    bo_error_buf[sizeof(bo_error_buf) - 1] = '\0';
}

const char *bo_get_error(void) {
    return bo_error_buf;
}

/* ========== PRNG extensions ========== */

double bo_rng_normal(bo_rng_t *rng) {
    /* Box-Muller */
    double u1 = bo_rng_uniform(rng);
    double u2 = bo_rng_uniform(rng);
    if (u1 < 1e-15) u1 = 1e-15;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

double bo_rng_gamma(bo_rng_t *rng, double shape) {
    /* Marsaglia-Tsang method for shape >= 1 */
    if (shape < 1.0) {
        /* Boost: Gamma(a) = Gamma(a+1) * U^(1/a) */
        double g = bo_rng_gamma(rng, shape + 1.0);
        double u = bo_rng_uniform(rng);
        if (u < 1e-15) u = 1e-15;
        return g * pow(u, 1.0 / shape);
    }
    double d = shape - 1.0 / 3.0;
    double c = 1.0 / sqrt(9.0 * d);
    for (;;) {
        double x, v;
        do {
            x = bo_rng_normal(rng);
            v = 1.0 + c * x;
        } while (v <= 0.0);
        v = v * v * v;
        double u = bo_rng_uniform(rng);
        if (u < 1.0 - 0.0331 * (x * x) * (x * x))
            return d * v;
        if (log(u) < 0.5 * x * x + d * (1.0 - v + log(v)))
            return d * v;
    }
}

double bo_rng_beta(bo_rng_t *rng, double a, double b) {
    double ga = bo_rng_gamma(rng, a);
    double gb = bo_rng_gamma(rng, b);
    double sum = ga + gb;
    if (sum < 1e-30) return 0.5;
    return ga / sum;
}

/* ========== Special math ========== */

double bo_norm_pdf(double x) {
    return exp(-0.5 * x * x) / sqrt(2.0 * M_PI);
}

double bo_norm_cdf(double x) {
    /* Abramowitz & Stegun 26.2.17 rational approximation */
    if (x < -8.0) return 0.0;
    if (x > 8.0) return 1.0;
    int neg = (x < 0.0);
    if (neg) x = -x;
    double t = 1.0 / (1.0 + 0.2316419 * x);
    double t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
    double poly = 0.319381530 * t - 0.356563782 * t2
                + 1.781477937 * t3 - 1.821255978 * t4
                + 1.330274429 * t5;
    double cdf = 1.0 - bo_norm_pdf(x) * poly;
    return neg ? 1.0 - cdf : cdf;
}

/* ========== Linear algebra ========== */

int bo_cholesky(double *A, int32_t n) {
    /* In-place col-major lower Cholesky: A = L L^T */
    for (int32_t j = 0; j < n; j++) {
        double sum = A[j * n + j];
        for (int32_t k = 0; k < j; k++) {
            double ljk = A[k * n + j];
            sum -= ljk * ljk;
        }
        if (sum <= 0.0) return -1;
        double ljj = sqrt(sum);
        A[j * n + j] = ljj;
        for (int32_t i = j + 1; i < n; i++) {
            sum = A[j * n + i];
            for (int32_t k = 0; k < j; k++)
                sum -= A[k * n + i] * A[k * n + j];
            A[j * n + i] = sum / ljj;
        }
        /* Zero upper triangle */
        for (int32_t i = 0; i < j; i++)
            A[j * n + i] = 0.0;
    }
    return 0;
}

double bo_cholesky_jitter(double *A, int32_t n) {
    /* Try adaptive jitter: 1e-8, 1e-7, ..., 1e-3 */
    double *backup = (double *)malloc((size_t)n * n * sizeof(double));
    if (!backup) return -1.0;
    memcpy(backup, A, (size_t)n * n * sizeof(double));

    double jitter = 1e-8;
    for (int attempt = 0; attempt < 6; attempt++) {
        memcpy(A, backup, (size_t)n * n * sizeof(double));
        for (int32_t i = 0; i < n; i++)
            A[i * n + i] += jitter;
        if (bo_cholesky(A, n) == 0) {
            free(backup);
            return jitter;
        }
        jitter *= 10.0;
    }
    free(backup);
    return -1.0;
}

void bo_trisolve_lower(const double *L, int32_t n, double *b) {
    /* Forward substitution: L x = b, col-major */
    for (int32_t j = 0; j < n; j++) {
        b[j] /= L[j * n + j];
        for (int32_t i = j + 1; i < n; i++)
            b[i] -= L[j * n + i] * b[j];
    }
}

void bo_trisolve_upper(const double *L, int32_t n, double *b) {
    /* Backward substitution: L^T x = b, L col-major */
    for (int32_t j = n - 1; j >= 0; j--) {
        for (int32_t i = j + 1; i < n; i++)
            b[j] -= L[j * n + i] * b[i];
        b[j] /= L[j * n + j];
    }
}

double bo_logdet_chol(const double *L, int32_t n) {
    double sum = 0.0;
    for (int32_t i = 0; i < n; i++)
        sum += log(L[i * n + i]);
    return 2.0 * sum;
}

/* ========== Kernels ========== */

/* Scaled distance: sqrt(sum_d ((x1_d - x2_d) / l_d)^2) */
static double scaled_dist(const double *x1, const double *x2,
                          int32_t dim, const double *ls) {
    double sum = 0.0;
    for (int32_t d = 0; d < dim; d++) {
        double diff = (x1[d] - x2[d]) / ls[d];
        sum += diff * diff;
    }
    return sqrt(sum);
}

double bo_kernel_matern52(const double *x1, const double *x2,
                          int32_t dim, const bo_hypers_t *h) {
    double r = scaled_dist(x1, x2, dim, h->lengthscales);
    double s5r = sqrt(5.0) * r;
    return h->signal_var * (1.0 + s5r + 5.0 / 3.0 * r * r) * exp(-s5r);
}

double bo_kernel_matern32(const double *x1, const double *x2,
                          int32_t dim, const bo_hypers_t *h) {
    double r = scaled_dist(x1, x2, dim, h->lengthscales);
    double s3r = sqrt(3.0) * r;
    return h->signal_var * (1.0 + s3r) * exp(-s3r);
}

double bo_kernel_se(const double *x1, const double *x2,
                    int32_t dim, const bo_hypers_t *h) {
    double r2 = 0.0;
    for (int32_t d = 0; d < dim; d++) {
        double diff = (x1[d] - x2[d]) / h->lengthscales[d];
        r2 += diff * diff;
    }
    return h->signal_var * exp(-0.5 * r2);
}

typedef double (*kernel_fn)(const double *, const double *,
                            int32_t, const bo_hypers_t *);

static kernel_fn get_kernel_fn(int32_t type) {
    switch (type) {
        case BO_KERNEL_MATERN52: return bo_kernel_matern52;
        case BO_KERNEL_MATERN32: return bo_kernel_matern32;
        case BO_KERNEL_SE:       return bo_kernel_se;
        default:                 return bo_kernel_matern52;
    }
}

void bo_kernel_matrix(const double *X, int32_t n, int32_t dim,
                      const bo_hypers_t *h, int32_t kernel_type,
                      double *K) {
    kernel_fn kfn = get_kernel_fn(kernel_type);
    for (int32_t i = 0; i < n; i++) {
        for (int32_t j = i; j < n; j++) {
            double k = kfn(X + i * dim, X + j * dim, dim, h);
            K[j * n + i] = k;  /* col-major */
            K[i * n + j] = k;
        }
    }
}

void bo_kernel_vector(const double *x_star, const double *X_train,
                      int32_t n, int32_t dim,
                      const bo_hypers_t *h, int32_t kernel_type,
                      double *k_vec) {
    kernel_fn kfn = get_kernel_fn(kernel_type);
    for (int32_t i = 0; i < n; i++)
        k_vec[i] = kfn(x_star, X_train + i * dim, dim, h);
}

/* Kernel gradient w.r.t. log-hyperparams.
 * Layout: dK[p * n*n + col*n + row] for hyperparam p.
 * Hyperparams: p=0..dim-1 = log-lengthscales, p=dim = log-signal, p=dim+1 = log-noise */
void bo_kernel_matrix_grad(const double *X, int32_t n, int32_t dim,
                           const bo_hypers_t *h, int32_t kernel_type,
                           double *dK) {
    int32_t nh = dim + 2;
    int32_t nn = n * n;
    memset(dK, 0, (size_t)nh * nn * sizeof(double));

    for (int32_t i = 0; i < n; i++) {
        for (int32_t j = i; j < n; j++) {
            const double *xi = X + i * dim;
            const double *xj = X + j * dim;

            /* Compute per-dimension squared scaled differences */
            double r2 = 0.0;
            for (int32_t d = 0; d < dim; d++) {
                double diff = (xi[d] - xj[d]) / h->lengthscales[d];
                r2 += diff * diff;
            }
            double r = sqrt(r2 + 1e-30);

            double k_val, dk_dr;
            if (kernel_type == BO_KERNEL_SE) {
                k_val = h->signal_var * exp(-0.5 * r2);
                dk_dr = -r * k_val;  /* dk/dr = -r * k */
            } else if (kernel_type == BO_KERNEL_MATERN32) {
                double s3r = sqrt(3.0) * r;
                double e = exp(-s3r);
                k_val = h->signal_var * (1.0 + s3r) * e;
                dk_dr = -3.0 * r * h->signal_var * e;  /* dk/dr */
            } else {
                /* Matern 5/2 */
                double s5r = sqrt(5.0) * r;
                double e = exp(-s5r);
                k_val = h->signal_var * (1.0 + s5r + 5.0 / 3.0 * r2) * e;
                dk_dr = -5.0 / 3.0 * r * (1.0 + s5r) * h->signal_var * e;
            }

            /* dK/d(log l_d) = dk/dr * dr/d(log l_d) = dk/dr * (-(x_d-x_d')^2 / (l_d^2 * r)) */
            for (int32_t d = 0; d < dim; d++) {
                double diff = xi[d] - xj[d];
                double ls_d = h->lengthscales[d];
                /* dr/d(log l_d) = -diff^2 / (l_d^2 * r)  (chain rule: d/d(log l) = l * d/dl) */
                double dr_dlog = -(diff * diff) / (ls_d * ls_d * (r + 1e-30));
                double val = dk_dr * dr_dlog;
                dK[d * nn + j * n + i] = val;
                dK[d * nn + i * n + j] = val;
            }

            /* dK/d(log signal_var) = k_val (since k = sv * f(r), dk/d(log sv) = sv * f(r) = k) */
            dK[dim * nn + j * n + i] = k_val;
            dK[dim * nn + i * n + j] = k_val;

            /* dK/d(log noise_var): only diagonal */
            if (i == j) {
                dK[(dim + 1) * nn + i * n + i] = h->noise_var;
            }
        }
    }
}

/* ========== GP ========== */

int bo_gp_fit(bo_gp_t *gp, const double *X, const double *y,
              int32_t n, int32_t dim,
              const bo_hypers_t *hypers, int32_t kernel_type) {
    gp->n = n;
    gp->dim = dim;
    gp->kernel = kernel_type;

    /* Copy hyperparameters */
    gp->hypers.signal_var = hypers->signal_var;
    gp->hypers.noise_var = hypers->noise_var;
    gp->hypers.lengthscales = (double *)malloc((size_t)dim * sizeof(double));
    if (!gp->hypers.lengthscales) { bo_set_error("alloc lengthscales"); return -1; }
    memcpy(gp->hypers.lengthscales, hypers->lengthscales, (size_t)dim * sizeof(double));

    /* Copy X */
    gp->X = (double *)malloc((size_t)n * dim * sizeof(double));
    if (!gp->X) { bo_set_error("alloc X"); return -1; }
    memcpy(gp->X, X, (size_t)n * dim * sizeof(double));

    /* Standardize y */
    gp->y_mean = 0.0;
    for (int32_t i = 0; i < n; i++) gp->y_mean += y[i];
    gp->y_mean /= n;

    gp->y_std = 0.0;
    for (int32_t i = 0; i < n; i++) {
        double d = y[i] - gp->y_mean;
        gp->y_std += d * d;
    }
    gp->y_std = sqrt(gp->y_std / n);
    if (gp->y_std < 1e-10) gp->y_std = 1.0;  /* constant y */

    gp->y = (double *)malloc((size_t)n * sizeof(double));
    if (!gp->y) { bo_set_error("alloc y"); return -1; }
    for (int32_t i = 0; i < n; i++)
        gp->y[i] = (y[i] - gp->y_mean) / gp->y_std;

    /* Build kernel matrix K + noise*I */
    gp->L = (double *)malloc((size_t)n * n * sizeof(double));
    if (!gp->L) { bo_set_error("alloc L"); return -1; }
    bo_kernel_matrix(X, n, dim, &gp->hypers, kernel_type, gp->L);
    for (int32_t i = 0; i < n; i++)
        gp->L[i * n + i] += gp->hypers.noise_var;

    /* Cholesky with adaptive jitter */
    double jitter = bo_cholesky_jitter(gp->L, n);
    if (jitter < 0.0) {
        bo_set_error("Cholesky failed: kernel matrix not positive definite");
        return -1;
    }

    /* alpha = L^T \ (L \ y) */
    gp->alpha = (double *)malloc((size_t)n * sizeof(double));
    if (!gp->alpha) { bo_set_error("alloc alpha"); return -1; }
    memcpy(gp->alpha, gp->y, (size_t)n * sizeof(double));
    bo_trisolve_lower(gp->L, n, gp->alpha);
    bo_trisolve_upper(gp->L, n, gp->alpha);

    /* Cache LML */
    gp->lml = bo_gp_lml(gp);

    return 0;
}

void bo_gp_predict(const bo_gp_t *gp, const double *x_star,
                   double *mean, double *var) {
    int32_t n = gp->n, dim = gp->dim;

    /* k* = kernel between x_star and training points */
    double *k_star = (double *)malloc((size_t)n * sizeof(double));
    bo_kernel_vector(x_star, gp->X, n, dim, &gp->hypers, gp->kernel, k_star);

    /* Predictive mean (standardized): k*^T alpha */
    double mu = 0.0;
    for (int32_t i = 0; i < n; i++)
        mu += k_star[i] * gp->alpha[i];

    /* Predictive variance: k** - k*^T K^{-1} k* = k** - v^T v where L v = k* */
    double *v = (double *)malloc((size_t)n * sizeof(double));
    memcpy(v, k_star, (size_t)n * sizeof(double));
    bo_trisolve_lower(gp->L, n, v);

    double k_ss = gp->hypers.signal_var;  /* k(x*, x*) */
    double var_std = k_ss;
    for (int32_t i = 0; i < n; i++)
        var_std -= v[i] * v[i];
    if (var_std < 0.0) var_std = 0.0;

    /* Unstandardize */
    *mean = mu * gp->y_std + gp->y_mean;
    *var = var_std * gp->y_std * gp->y_std;

    free(k_star);
    free(v);
}

double bo_gp_lml(const bo_gp_t *gp) {
    int32_t n = gp->n;
    double data_fit = 0.0;
    for (int32_t i = 0; i < n; i++)
        data_fit += gp->y[i] * gp->alpha[i];
    double logdet = bo_logdet_chol(gp->L, n);
    return -0.5 * (data_fit + logdet + n * log(2.0 * M_PI));
}

void bo_gp_lml_grad(const bo_gp_t *gp, const double *X,
                    double *grad) {
    int32_t n = gp->n, dim = gp->dim;
    int32_t nh = dim + 2;

    /* Compute K^{-1} via Cholesky: solve L L^T K_inv_col = e_i for each i
     * But we only need alpha * alpha^T - K^{-1}, so compute:
     * For each hyperparam p: grad[p] = 0.5 * tr((alpha alpha^T - K^{-1}) dK_p)
     *
     * Efficiently: grad[p] = 0.5 * (alpha^T dK_p alpha - tr(K^{-1} dK_p))
     * For tr(K^{-1} dK_p), we use: K^{-1} = L^{-T} L^{-1}, so
     * tr(K^{-1} dK_p) = sum_{ij} (L^{-1})_{ij} (L^{-1} dK_p)_{ij}
     * which requires computing L^{-1} explicitly -- expensive.
     *
     * Alternative: compute K_inv columns one at a time.
     */
    double *dK = (double *)malloc((size_t)nh * n * n * sizeof(double));
    if (!dK) return;
    bo_kernel_matrix_grad(X, n, dim, &gp->hypers, gp->kernel, dK);

    /* Compute K^{-1} column by column using Cholesky */
    double *Kinv = (double *)calloc((size_t)n * n, sizeof(double));
    if (!Kinv) { free(dK); return; }
    double *col = (double *)malloc((size_t)n * sizeof(double));
    if (!col) { free(dK); free(Kinv); return; }

    for (int32_t j = 0; j < n; j++) {
        memset(col, 0, (size_t)n * sizeof(double));
        col[j] = 1.0;
        bo_trisolve_lower(gp->L, n, col);
        bo_trisolve_upper(gp->L, n, col);
        for (int32_t i = 0; i < n; i++)
            Kinv[j * n + i] = col[i];
    }
    free(col);

    /* grad[p] = 0.5 * tr((alpha alpha^T - K^{-1}) dK_p) */
    for (int32_t p = 0; p < nh; p++) {
        double *dKp = dK + (size_t)p * n * n;
        double sum = 0.0;
        for (int32_t i = 0; i < n; i++) {
            for (int32_t j = 0; j < n; j++) {
                double aat = gp->alpha[i] * gp->alpha[j];
                double kinv = Kinv[j * n + i];
                sum += (aat - kinv) * dKp[j * n + i];
            }
        }
        grad[p] = 0.5 * sum;
    }

    free(dK);
    free(Kinv);
}

void bo_gp_free(bo_gp_t *gp) {
    free(gp->X);      gp->X = NULL;
    free(gp->y);      gp->y = NULL;
    free(gp->L);      gp->L = NULL;
    free(gp->alpha);   gp->alpha = NULL;
    free(gp->hypers.lengthscales); gp->hypers.lengthscales = NULL;
    gp->n = 0;
    gp->dim = 0;
}

/* ========== L-BFGS ========== */

/* Minimal bounded L-BFGS implementation */
#define LBFGS_M 5

double bo_lbfgs_minimize(bo_obj_fn fn, void *ctx,
                         double *x, int32_t n,
                         const double *lower, const double *upper,
                         int32_t max_iter, double ftol) {
    if (n <= 0) return 0.0;

    double *grad = (double *)malloc((size_t)n * sizeof(double));
    double *x_prev = (double *)malloc((size_t)n * sizeof(double));
    double *g_prev = (double *)malloc((size_t)n * sizeof(double));
    double *dir = (double *)malloc((size_t)n * sizeof(double));
    /* L-BFGS history */
    double *S = (double *)calloc((size_t)LBFGS_M * n, sizeof(double));
    double *Y = (double *)calloc((size_t)LBFGS_M * n, sizeof(double));
    double *rho = (double *)calloc(LBFGS_M, sizeof(double));
    double *alpha_buf = (double *)malloc((size_t)LBFGS_M * sizeof(double));

    if (!grad || !x_prev || !g_prev || !dir || !S || !Y || !rho || !alpha_buf) {
        free(grad); free(x_prev); free(g_prev); free(dir);
        free(S); free(Y); free(rho); free(alpha_buf);
        return 1e30;
    }

    /* Project x onto bounds */
    if (lower && upper) {
        for (int32_t i = 0; i < n; i++) {
            if (x[i] < lower[i]) x[i] = lower[i];
            if (x[i] > upper[i]) x[i] = upper[i];
        }
    }

    double f = fn(x, grad, n, ctx);
    int32_t k = 0;

    for (int32_t iter = 0; iter < max_iter; iter++) {
        /* Project gradient for bounded variables */
        if (lower && upper) {
            for (int32_t i = 0; i < n; i++) {
                if (x[i] <= lower[i] && grad[i] > 0.0) grad[i] = 0.0;
                if (x[i] >= upper[i] && grad[i] < 0.0) grad[i] = 0.0;
            }
        }

        /* Check convergence */
        double gnorm = 0.0;
        for (int32_t i = 0; i < n; i++) gnorm += grad[i] * grad[i];
        gnorm = sqrt(gnorm);
        if (gnorm < ftol) break;

        /* Save previous */
        memcpy(x_prev, x, (size_t)n * sizeof(double));
        memcpy(g_prev, grad, (size_t)n * sizeof(double));

        /* L-BFGS two-loop recursion to compute direction */
        memcpy(dir, grad, (size_t)n * sizeof(double));
        int32_t m_use = k < LBFGS_M ? k : LBFGS_M;

        for (int32_t i = m_use - 1; i >= 0; i--) {
            int32_t idx = (k - 1 - (m_use - 1 - i)) % LBFGS_M;
            if (idx < 0) idx += LBFGS_M;
            double dot = 0.0;
            for (int32_t j = 0; j < n; j++)
                dot += S[idx * n + j] * dir[j];
            alpha_buf[i] = rho[idx] * dot;
            for (int32_t j = 0; j < n; j++)
                dir[j] -= alpha_buf[i] * Y[idx * n + j];
        }

        /* Scale by H0 = (s^T y / y^T y) I */
        if (m_use > 0) {
            int32_t last = (k - 1) % LBFGS_M;
            if (last < 0) last += LBFGS_M;
            double sy = 0.0, yy = 0.0;
            for (int32_t j = 0; j < n; j++) {
                sy += S[last * n + j] * Y[last * n + j];
                yy += Y[last * n + j] * Y[last * n + j];
            }
            double gamma = (yy > 1e-30) ? sy / yy : 1.0;
            for (int32_t j = 0; j < n; j++) dir[j] *= gamma;
        }

        for (int32_t i = 0; i < m_use; i++) {
            int32_t idx = (k - m_use + i) % LBFGS_M;
            if (idx < 0) idx += LBFGS_M;
            double dot = 0.0;
            for (int32_t j = 0; j < n; j++)
                dot += Y[idx * n + j] * dir[j];
            double beta_val = rho[idx] * dot;
            for (int32_t j = 0; j < n; j++)
                dir[j] += (alpha_buf[i] - beta_val) * S[idx * n + j];
        }

        /* Negate for descent */
        for (int32_t i = 0; i < n; i++) dir[i] = -dir[i];

        /* Backtracking line search */
        double step = 1.0;
        double f_new;
        double dg = 0.0;
        for (int32_t i = 0; i < n; i++) dg += grad[i] * dir[i];

        double f_prev = f;
        for (int32_t ls = 0; ls < 20; ls++) {
            for (int32_t i = 0; i < n; i++)
                x[i] = x_prev[i] + step * dir[i];
            /* Project onto bounds */
            if (lower && upper) {
                for (int32_t i = 0; i < n; i++) {
                    if (x[i] < lower[i]) x[i] = lower[i];
                    if (x[i] > upper[i]) x[i] = upper[i];
                }
            }
            f_new = fn(x, grad, n, ctx);
            if (f_new <= f + 1e-4 * step * dg || step < 1e-10) {
                break;
            }
            step *= 0.5;
        }
        f = f_new;

        /* Update L-BFGS history */
        int32_t slot = k % LBFGS_M;
        double sy = 0.0, yy_val = 0.0;
        for (int32_t i = 0; i < n; i++) {
            S[slot * n + i] = x[i] - x_prev[i];
            Y[slot * n + i] = grad[i] - g_prev[i];
            sy += S[slot * n + i] * Y[slot * n + i];
            yy_val += Y[slot * n + i] * Y[slot * n + i];
        }
        rho[slot] = (fabs(sy) > 1e-30) ? 1.0 / sy : 0.0;
        k++;

        /* Check function change convergence */
        if (fabs(f_new - f_prev) < ftol * (1.0 + fabs(f_prev))) break;
    }

    free(grad); free(x_prev); free(g_prev); free(dir);
    free(S); free(Y); free(rho); free(alpha_buf);
    return f;
}

/* ========== Acquisition functions ========== */

double bo_acq_ei(double mu, double sigma, double f_best, double xi) {
    if (sigma < 1e-12) return 0.0;
    double z = (mu - f_best - xi) / sigma;
    return (mu - f_best - xi) * bo_norm_cdf(z) + sigma * bo_norm_pdf(z);
}

double bo_acq_ucb(double mu, double sigma, double kappa) {
    return mu + kappa * sigma;
}

double bo_acq_pi(double mu, double sigma, double f_best, double xi) {
    if (sigma < 1e-12) return 0.0;
    double z = (mu - f_best - xi) / sigma;
    return bo_norm_cdf(z);
}

/* Context for acquisition optimization */
typedef struct {
    const bo_gp_t *gp;
    int32_t acq_fn;
    double kappa, xi, f_best;
} acq_ctx_t;

double bo_acq_eval_grad(const bo_gp_t *gp, const double *x,
                        int32_t acq_fn, double kappa, double xi,
                        double f_best, double *grad) {
    double mu, var;
    bo_gp_predict(gp, x, &mu, &var);
    double sigma = sqrt(var + 1e-30);

    double acq;
    if (acq_fn == BO_ACQ_EI) {
        acq = bo_acq_ei(mu, sigma, f_best, xi);
    } else if (acq_fn == BO_ACQ_UCB) {
        acq = bo_acq_ucb(mu, sigma, kappa);
    } else {
        acq = bo_acq_pi(mu, sigma, f_best, xi);
    }

    if (grad) {
        /* Numerical gradient (simple, robust) */
        double eps = 1e-5;
        int32_t dim = gp->dim;
        double *x_pert = (double *)malloc((size_t)dim * sizeof(double));
        memcpy(x_pert, x, (size_t)dim * sizeof(double));
        for (int32_t d = 0; d < dim; d++) {
            double orig = x_pert[d];
            x_pert[d] = orig + eps;
            double mu_p, var_p;
            bo_gp_predict(gp, x_pert, &mu_p, &var_p);
            double sigma_p = sqrt(var_p + 1e-30);
            double acq_p;
            if (acq_fn == BO_ACQ_EI)
                acq_p = bo_acq_ei(mu_p, sigma_p, f_best, xi);
            else if (acq_fn == BO_ACQ_UCB)
                acq_p = bo_acq_ucb(mu_p, sigma_p, kappa);
            else
                acq_p = bo_acq_pi(mu_p, sigma_p, f_best, xi);
            /* We minimize negative acquisition */
            grad[d] = -(acq_p - acq) / eps;
            x_pert[d] = orig;
        }
        free(x_pert);
    }

    return -acq;  /* negate for minimization */
}

static double acq_objective(const double *x, double *grad, int32_t n, void *ctx_) {
    acq_ctx_t *ctx = (acq_ctx_t *)ctx_;
    return bo_acq_eval_grad(ctx->gp, x, ctx->acq_fn,
                            ctx->kappa, ctx->xi, ctx->f_best, grad);
}

double bo_optimize_acquisition(const bo_gp_t *gp,
                               int32_t acq_fn, double kappa, double xi,
                               double f_best,
                               int32_t n_restarts, bo_rng_t *rng,
                               double *x_best) {
    int32_t dim = gp->dim;
    if (dim <= 0) return 0.0;

    double *lower = (double *)malloc((size_t)dim * sizeof(double));
    double *upper = (double *)malloc((size_t)dim * sizeof(double));
    double *x_try = (double *)malloc((size_t)dim * sizeof(double));
    for (int32_t d = 0; d < dim; d++) {
        lower[d] = 0.0;
        upper[d] = 1.0;
    }

    acq_ctx_t ctx = { gp, acq_fn, kappa, xi, f_best };
    double best_val = 1e30;
    int found_finite = 0;

    for (int32_t r = 0; r < n_restarts; r++) {
        for (int32_t d = 0; d < dim; d++)
            x_try[d] = bo_rng_uniform(rng);

        double val = bo_lbfgs_minimize(acq_objective, &ctx,
                                       x_try, dim, lower, upper, 50, 1e-6);
        if (!isfinite(val)) continue;
        if (!found_finite || val < best_val) {
            found_finite = 1;
            best_val = val;
            memcpy(x_best, x_try, (size_t)dim * sizeof(double));
        }
    }

    if (!found_finite) {
        for (int32_t d = 0; d < dim; d++)
            x_best[d] = bo_rng_uniform(rng);
        best_val = acq_objective(x_best, NULL, dim, &ctx);
        if (!isfinite(best_val)) best_val = 0.0;
    }

    free(lower); free(upper); free(x_try);
    return -best_val;  /* return positive acquisition value */
}

/* ========== Hyperparameter optimization ========== */

typedef struct {
    const double *X;
    const double *y;
    int32_t n, dim, kernel;
} hyper_ctx_t;

static double hyper_objective(const double *log_hypers, double *grad,
                              int32_t nh, void *ctx_) {
    hyper_ctx_t *ctx = (hyper_ctx_t *)ctx_;
    int32_t n = ctx->n, dim = ctx->dim;

    /* Decode hyperparams from log space */
    bo_hypers_t h;
    h.lengthscales = (double *)malloc((size_t)dim * sizeof(double));
    for (int32_t d = 0; d < dim; d++)
        h.lengthscales[d] = exp(log_hypers[d]);
    h.signal_var = exp(log_hypers[dim]);
    h.noise_var = exp(log_hypers[dim + 1]);

    /* Fit GP */
    bo_gp_t gp;
    memset(&gp, 0, sizeof(gp));
    int ret = bo_gp_fit(&gp, ctx->X, ctx->y, n, dim, &h, ctx->kernel);
    free(h.lengthscales);

    if (ret != 0) {
        bo_gp_free(&gp);
        if (grad) memset(grad, 0, (size_t)nh * sizeof(double));
        return 1e20;
    }

    double neg_lml = -gp.lml;

    if (grad) {
        double *lml_grad = (double *)malloc((size_t)nh * sizeof(double));
        bo_gp_lml_grad(&gp, ctx->X, lml_grad);
        /* Negate and apply chain rule: d/d(log_h) = d/dh * h = grad * exp(log_h) */
        /* Actually, bo_gp_lml_grad already gives grad w.r.t. log-hypers (the kernel
         * grad is w.r.t. log-params), so just negate. */
        for (int32_t p = 0; p < nh; p++)
            grad[p] = -lml_grad[p];
        free(lml_grad);
    }

    bo_gp_free(&gp);
    return neg_lml;
}

double bo_optimize_hypers(bo_gp_t *gp, const double *X, const double *y,
                          int32_t n, int32_t dim,
                          int32_t n_restarts, bo_rng_t *rng) {
    int32_t nh = dim + 2;
    double *lower = (double *)malloc((size_t)nh * sizeof(double));
    double *upper = (double *)malloc((size_t)nh * sizeof(double));
    double *log_h = (double *)malloc((size_t)nh * sizeof(double));
    double *best_log_h = (double *)malloc((size_t)nh * sizeof(double));

    for (int32_t i = 0; i < nh; i++) {
        lower[i] = -5.0;
        upper[i] = 5.0;
    }
    /* Tighter bounds for noise */
    lower[dim + 1] = -8.0;
    upper[dim + 1] = 0.0;

    hyper_ctx_t ctx = { X, y, n, dim, gp->kernel };
    double best_val = 1e30;
    int found_finite = 0;

    for (int32_t r = 0; r < n_restarts; r++) {
        if (r == 0 && gp->hypers.lengthscales) {
            /* First restart: use current hyperparams */
            for (int32_t d = 0; d < dim; d++)
                log_h[d] = log(gp->hypers.lengthscales[d]);
            log_h[dim] = log(gp->hypers.signal_var);
            log_h[dim + 1] = log(gp->hypers.noise_var);
        } else {
            /* Random restart */
            for (int32_t i = 0; i < nh; i++)
                log_h[i] = lower[i] + bo_rng_uniform(rng) * (upper[i] - lower[i]);
        }

        double val = bo_lbfgs_minimize(hyper_objective, &ctx,
                                       log_h, nh, lower, upper, 100, 1e-6);
        if (!isfinite(val)) continue;
        if (!found_finite || val < best_val) {
            found_finite = 1;
            best_val = val;
            memcpy(best_log_h, log_h, (size_t)nh * sizeof(double));
        }
    }

    if (!found_finite) {
        if (gp->hypers.lengthscales) {
            for (int32_t d = 0; d < dim; d++)
                best_log_h[d] = log(gp->hypers.lengthscales[d]);
            best_log_h[dim] = log(gp->hypers.signal_var);
            best_log_h[dim + 1] = log(gp->hypers.noise_var);
        } else {
            for (int32_t i = 0; i < nh; i++)
                best_log_h[i] = 0.0;
            best_log_h[dim + 1] = -4.0;
        }
        best_val = hyper_objective(best_log_h, NULL, nh, &ctx);
        if (!isfinite(best_val)) best_val = 0.0;
    }

    /* Refit with best hypers */
    bo_hypers_t best_h;
    best_h.lengthscales = (double *)malloc((size_t)dim * sizeof(double));
    if (!best_h.lengthscales) {
        free(lower); free(upper); free(log_h); free(best_log_h);
        return -best_val;
    }
    for (int32_t d = 0; d < dim; d++)
        best_h.lengthscales[d] = exp(best_log_h[d]);
    best_h.signal_var = exp(best_log_h[dim]);
    best_h.noise_var = exp(best_log_h[dim + 1]);

    bo_gp_free(gp);
    bo_gp_fit(gp, X, y, n, dim, &best_h, gp->kernel);
    free(best_h.lengthscales);

    free(lower); free(upper); free(log_h); free(best_log_h);
    return -best_val;  /* return best LML */
}

/* ========== Search space ========== */

bo_space_t *bo_space_create(int32_t n_dims) {
    bo_space_t *s = (bo_space_t *)calloc(1, sizeof(bo_space_t));
    if (!s) return NULL;
    s->dims = (bo_dim_desc_t *)calloc((size_t)n_dims, sizeof(bo_dim_desc_t));
    if (!s->dims) { free(s); return NULL; }
    s->n_dims = n_dims;
    for (int32_t i = 0; i < n_dims; i++) {
        s->dims[i].cond_parent = -1;
        s->dims[i].id = i;
    }
    return s;
}

int bo_space_add_continuous(bo_space_t *s, int32_t id,
                            double low, double high, int32_t log_scale) {
    if (id < 0 || id >= s->n_dims) return -1;
    s->dims[id].id = id;
    s->dims[id].type = BO_DIM_CONTINUOUS;
    s->dims[id].low = low;
    s->dims[id].high = high;
    s->dims[id].log_scale = log_scale;
    s->dims[id].n_values = 0;
    return 0;
}

int bo_space_add_integer(bo_space_t *s, int32_t id,
                         double low, double high, int32_t log_scale) {
    if (id < 0 || id >= s->n_dims) return -1;
    s->dims[id].id = id;
    s->dims[id].type = BO_DIM_INTEGER;
    s->dims[id].low = low;
    s->dims[id].high = high;
    s->dims[id].log_scale = log_scale;
    s->dims[id].n_values = 0;
    return 0;
}

int bo_space_add_categorical(bo_space_t *s, int32_t id, int32_t n_values) {
    if (id < 0 || id >= s->n_dims) return -1;
    s->dims[id].id = id;
    s->dims[id].type = BO_DIM_CATEGORICAL;
    s->dims[id].n_values = n_values;
    return 0;
}

int bo_space_add_condition(bo_space_t *s, int32_t id,
                           int32_t parent_id, int32_t parent_value) {
    if (id < 0 || id >= s->n_dims) return -1;
    s->dims[id].cond_parent = parent_id;
    s->dims[id].cond_value = parent_value;
    return 0;
}

void bo_space_finalize(bo_space_t *s) {
    /* Count and index categorical vs continuous dims */
    s->n_categorical = 0;
    s->n_continuous = 0;
    for (int32_t i = 0; i < s->n_dims; i++) {
        if (s->dims[i].type == BO_DIM_CATEGORICAL)
            s->n_categorical++;
        else
            s->n_continuous++;
    }
    free(s->cat_ids); free(s->cont_ids);
    s->cat_ids = (int32_t *)malloc((size_t)s->n_categorical * sizeof(int32_t));
    s->cont_ids = (int32_t *)malloc((size_t)s->n_continuous * sizeof(int32_t));
    int32_t ci = 0, ki = 0;
    for (int32_t i = 0; i < s->n_dims; i++) {
        if (s->dims[i].type == BO_DIM_CATEGORICAL)
            s->cat_ids[ki++] = i;
        else
            s->cont_ids[ci++] = i;
    }
    s->finalized = 1;
}

void bo_space_free(bo_space_t *s) {
    if (!s) return;
    free(s->dims);
    free(s->cat_ids);
    free(s->cont_ids);
    free(s);
}

int bo_space_is_active(const bo_space_t *s, int32_t dim_id,
                       const int32_t *cat_values) {
    const bo_dim_desc_t *d = &s->dims[dim_id];
    if (d->cond_parent < 0) return 1;
    /* Find parent's categorical value */
    int32_t parent = d->cond_parent;
    /* cat_values is indexed by position in cat_ids, but cond_parent is dim_id.
     * We need to find the parent's value from the full param vector. */
    /* Actually, cat_values here is [n_categorical] indexed by cat_ids position.
     * To check: find which cat_ids position has parent_id */
    for (int32_t k = 0; k < s->n_categorical; k++) {
        if (s->cat_ids[k] == parent)
            return (cat_values[k] == d->cond_value) ? 1 : 0;
    }
    return 0;  /* parent not found (shouldn't happen) */
}

int32_t bo_space_count_active(const bo_space_t *s, const int32_t *cat_values) {
    int32_t count = 0;
    for (int32_t i = 0; i < s->n_continuous; i++) {
        int32_t dim_id = s->cont_ids[i];
        if (bo_space_is_active(s, dim_id, cat_values))
            count++;
    }
    return count;
}

int32_t bo_space_encode(const bo_space_t *s, const double *raw,
                        const int32_t *cat_values, double *encoded) {
    int32_t idx = 0;
    for (int32_t i = 0; i < s->n_continuous; i++) {
        int32_t dim_id = s->cont_ids[i];
        if (!bo_space_is_active(s, dim_id, cat_values)) continue;
        const bo_dim_desc_t *d = &s->dims[dim_id];
        double v = raw[dim_id];
        if (d->log_scale) {
            double log_lo = log(d->low);
            double log_hi = log(d->high);
            encoded[idx] = (log(v) - log_lo) / (log_hi - log_lo);
        } else {
            encoded[idx] = (v - d->low) / (d->high - d->low);
        }
        /* Clamp to [0, 1] */
        if (encoded[idx] < 0.0) encoded[idx] = 0.0;
        if (encoded[idx] > 1.0) encoded[idx] = 1.0;
        idx++;
    }
    return idx;
}

void bo_space_decode(const bo_space_t *s, const double *encoded,
                     const int32_t *cat_values, double *raw,
                     int32_t n_active) {
    int32_t idx = 0;
    for (int32_t i = 0; i < s->n_continuous; i++) {
        int32_t dim_id = s->cont_ids[i];
        if (!bo_space_is_active(s, dim_id, cat_values)) continue;
        if (idx >= n_active) break;
        const bo_dim_desc_t *d = &s->dims[dim_id];
        double e = encoded[idx];
        if (e < 0.0) e = 0.0;
        if (e > 1.0) e = 1.0;
        if (d->log_scale) {
            double log_lo = log(d->low);
            double log_hi = log(d->high);
            raw[dim_id] = exp(log_lo + e * (log_hi - log_lo));
        } else {
            raw[dim_id] = d->low + e * (d->high - d->low);
        }
        if (d->type == BO_DIM_INTEGER)
            raw[dim_id] = round(raw[dim_id]);
        idx++;
    }
}

/* ========== Categorical Thompson sampling ========== */

void cat_sampler_init(bo_cat_sampler_t *cs, int32_t n_values) {
    cs->n_values = n_values;
    cs->alpha = (double *)malloc((size_t)n_values * sizeof(double));
    cs->beta = (double *)malloc((size_t)n_values * sizeof(double));
    for (int32_t i = 0; i < n_values; i++) {
        cs->alpha[i] = 1.0;  /* Beta(1,1) = Uniform prior */
        cs->beta[i] = 1.0;
    }
}

int32_t cat_sampler_sample(bo_cat_sampler_t *cs, bo_rng_t *rng) {
    int32_t best = 0;
    double best_val = -1.0;
    for (int32_t i = 0; i < cs->n_values; i++) {
        double sample = bo_rng_beta(rng, cs->alpha[i], cs->beta[i]);
        if (sample > best_val) {
            best_val = sample;
            best = i;
        }
    }
    return best;
}

void cat_sampler_observe(bo_cat_sampler_t *cs,
                                int32_t value, double score, double median) {
    if (value < 0 || value >= cs->n_values) return;
    if (score >= median)
        cs->alpha[value] += 1.0;
    else
        cs->beta[value] += 1.0;
}

void cat_sampler_free(bo_cat_sampler_t *cs) {
    free(cs->alpha); cs->alpha = NULL;
    free(cs->beta);  cs->beta = NULL;
}

/* ========== Context management ========== */

static int cat_key_eq(const int32_t *a, const int32_t *b, int32_t len) {
    for (int32_t i = 0; i < len; i++)
        if (a[i] != b[i]) return 0;
    return 1;
}

static bo_context_t *find_or_create_context(bo_state_t *state,
                                            const int32_t *cat_values) {
    int32_t nc = state->space.n_categorical;
    /* Search existing */
    for (int32_t i = 0; i < state->n_contexts; i++) {
        if (cat_key_eq(state->contexts[i].cat_key, cat_values, nc))
            return &state->contexts[i];
    }
    /* Create new */
    if (state->n_contexts >= state->ctx_cap) {
        int32_t new_cap = state->ctx_cap < 4 ? 8 : state->ctx_cap * 2;
        bo_context_t *new_ctx = (bo_context_t *)realloc(
            state->contexts, (size_t)new_cap * sizeof(bo_context_t));
        if (!new_ctx) return NULL;
        state->contexts = new_ctx;
        state->ctx_cap = new_cap;
    }
    bo_context_t *ctx = &state->contexts[state->n_contexts];
    memset(ctx, 0, sizeof(bo_context_t));
    ctx->cat_key = (int32_t *)malloc((size_t)nc * sizeof(int32_t));
    memcpy(ctx->cat_key, cat_values, (size_t)nc * sizeof(int32_t));
    state->n_contexts++;
    return ctx;
}

static void context_add_obs(bo_context_t *ctx, int32_t obs_idx) {
    if (ctx->n_obs >= ctx->obs_cap) {
        int32_t new_cap = ctx->obs_cap < 4 ? 8 : ctx->obs_cap * 2;
        ctx->obs_indices = (int32_t *)realloc(ctx->obs_indices,
                                               (size_t)new_cap * sizeof(int32_t));
        ctx->obs_cap = new_cap;
    }
    ctx->obs_indices[ctx->n_obs++] = obs_idx;
    ctx->gp_fitted = 0;  /* invalidate */
}

static void context_free(bo_context_t *ctx) {
    free(ctx->cat_key);
    free(ctx->obs_indices);
    bo_gp_free(&ctx->gp);
}

/* ========== Top-level BO ========== */

bo_state_t *bo_create(const bo_space_t *space,
                      int32_t kernel, int32_t acq_fn,
                      double kappa, double xi,
                      int32_t n_restarts_hyper, int32_t n_restarts_acq,
                      int32_t max_obs, int32_t min_obs_for_gp,
                      uint32_t seed) {
    bo_state_t *st = (bo_state_t *)calloc(1, sizeof(bo_state_t));
    if (!st) return NULL;

    /* Copy space */
    st->space.n_dims = space->n_dims;
    st->space.n_continuous = space->n_continuous;
    st->space.n_categorical = space->n_categorical;
    st->space.finalized = space->finalized;
    st->space.dims = (bo_dim_desc_t *)malloc(
        (size_t)space->n_dims * sizeof(bo_dim_desc_t));
    memcpy(st->space.dims, space->dims,
           (size_t)space->n_dims * sizeof(bo_dim_desc_t));
    st->space.cat_ids = (int32_t *)malloc(
        (size_t)space->n_categorical * sizeof(int32_t));
    memcpy(st->space.cat_ids, space->cat_ids,
           (size_t)space->n_categorical * sizeof(int32_t));
    st->space.cont_ids = (int32_t *)malloc(
        (size_t)space->n_continuous * sizeof(int32_t));
    memcpy(st->space.cont_ids, space->cont_ids,
           (size_t)space->n_continuous * sizeof(int32_t));

    st->rng.state = seed;
    st->kernel = kernel;
    st->acq_fn = acq_fn;
    st->kappa = kappa;
    st->xi = xi;
    st->n_restarts_hyper = n_restarts_hyper > 0 ? n_restarts_hyper : 5;
    st->n_restarts_acq = n_restarts_acq > 0 ? n_restarts_acq : 20;
    st->max_obs = max_obs > 0 ? max_obs : 500;
    st->min_obs_for_gp = min_obs_for_gp > 0 ? min_obs_for_gp : 3;

    /* Init categorical samplers */
    int32_t nc = space->n_categorical;
    if (nc > 0) {
        st->cat_samplers = (bo_cat_sampler_t *)calloc(
            (size_t)nc, sizeof(bo_cat_sampler_t));
        for (int32_t k = 0; k < nc; k++) {
            int32_t dim_id = space->cat_ids[k];
            cat_sampler_init(&st->cat_samplers[k], space->dims[dim_id].n_values);
        }
    }

    /* Pre-allocate observation storage */
    st->obs_cap = 32;
    st->obs = (bo_obs_t *)calloc((size_t)st->obs_cap, sizeof(bo_obs_t));

    return st;
}

int bo_observe(bo_state_t *state, const double *params, double score) {
    /* Grow if needed */
    if (state->n_obs >= state->obs_cap) {
        int32_t new_cap = state->obs_cap * 2;
        bo_obs_t *new_obs = (bo_obs_t *)realloc(
            state->obs, (size_t)new_cap * sizeof(bo_obs_t));
        if (!new_obs) { bo_set_error("alloc obs"); return -1; }
        state->obs = new_obs;
        /* Zero new entries */
        memset(state->obs + state->obs_cap, 0,
               (size_t)(new_cap - state->obs_cap) * sizeof(bo_obs_t));
        state->obs_cap = new_cap;
    }

    int32_t nd = state->space.n_dims;
    bo_obs_t *o = &state->obs[state->n_obs];
    o->values = (double *)malloc((size_t)nd * sizeof(double));
    if (!o->values) { bo_set_error("alloc obs values"); return -1; }
    memcpy(o->values, params, (size_t)nd * sizeof(double));
    o->score = score;

    /* Update running stats */
    state->score_sum += score;
    state->score_count += 1.0;
    double median_approx = state->score_sum / state->score_count;

    /* Update Thompson sampling */
    int32_t nc = state->space.n_categorical;
    int32_t *cat_vals = NULL;
    if (nc > 0) {
        cat_vals = (int32_t *)malloc((size_t)nc * sizeof(int32_t));
        for (int32_t k = 0; k < nc; k++) {
            int32_t dim_id = state->space.cat_ids[k];
            cat_vals[k] = (int32_t)params[dim_id];
            cat_sampler_observe(&state->cat_samplers[k],
                                cat_vals[k], score, median_approx);
        }
    }

    /* Add to context */
    if (nc > 0) {
        bo_context_t *ctx = find_or_create_context(state, cat_vals);
        if (ctx) context_add_obs(ctx, state->n_obs);
        free(cat_vals);
    } else {
        /* No categoricals: single context */
        if (state->n_contexts == 0) {
            state->contexts = (bo_context_t *)calloc(1, sizeof(bo_context_t));
            state->n_contexts = 1;
            state->ctx_cap = 1;
            state->contexts[0].cat_key = NULL;
        }
        context_add_obs(&state->contexts[0], state->n_obs);
    }

    state->n_obs++;

    /* Reservoir pruning if needed */
    if (state->n_obs > state->max_obs) {
        /* TODO: implement reservoir pruning (keep best half + recent half) */
        /* For now, just cap. Full impl in future iteration. */
    }

    return 0;
}

int bo_suggest(bo_state_t *state, double *params) {
    int32_t nd = state->space.n_dims;
    int32_t nc = state->space.n_categorical;

    /* Step 1: Select categorical values via Thompson sampling */
    int32_t *cat_vals = NULL;
    if (nc > 0) {
        cat_vals = (int32_t *)calloc((size_t)nc, sizeof(int32_t));
        for (int32_t k = 0; k < nc; k++)
            cat_vals[k] = cat_sampler_sample(&state->cat_samplers[k], &state->rng);
    }

    /* Set categorical values in output */
    for (int32_t k = 0; k < nc; k++) {
        int32_t dim_id = state->space.cat_ids[k];
        params[dim_id] = (double)cat_vals[k];
    }

    /* Step 2: Find context for this categorical combination */
    int32_t n_active = bo_space_count_active(&state->space, cat_vals);

    if (n_active == 0) {
        /* No continuous dims -- just categorical, done */
        free(cat_vals);
        return 0;
    }

    bo_context_t *ctx = NULL;
    if (nc > 0) {
        ctx = find_or_create_context(state, cat_vals);
    } else if (state->n_contexts > 0) {
        ctx = &state->contexts[0];
    }

    /* Step 3: If enough observations, use GP; otherwise random */
    if (ctx && ctx->n_obs >= state->min_obs_for_gp && n_active > 0) {
        /* Build training data for this context */
        int32_t n_train = ctx->n_obs;
        double *X_train = (double *)malloc((size_t)n_train * n_active * sizeof(double));
        double *y_train = (double *)malloc((size_t)n_train * sizeof(double));
        if (!X_train || !y_train) {
            free(X_train); free(y_train); free(cat_vals);
            bo_set_error("alloc train data");
            return -1;
        }

        for (int32_t i = 0; i < n_train; i++) {
            int32_t obs_idx = ctx->obs_indices[i];
            bo_obs_t *o = &state->obs[obs_idx];
            bo_space_encode(&state->space, o->values, cat_vals,
                            X_train + i * n_active);
            y_train[i] = o->score;
        }

        /* Init hyperparams if needed */
        bo_hypers_t init_h;
        init_h.lengthscales = (double *)malloc((size_t)n_active * sizeof(double));
        for (int32_t d = 0; d < n_active; d++)
            init_h.lengthscales[d] = 0.5;  /* reasonable default for [0,1] space */
        init_h.signal_var = 1.0;
        init_h.noise_var = 0.01;

        /* Fit GP */
        bo_gp_t gp;
        memset(&gp, 0, sizeof(gp));
        gp.kernel = state->kernel;
        int ret = bo_gp_fit(&gp, X_train, y_train, n_train, n_active,
                            &init_h, state->kernel);
        free(init_h.lengthscales);

        if (ret == 0) {
            /* Optimize hyperparameters */
            bo_optimize_hypers(&gp, X_train, y_train, n_train, n_active,
                               state->n_restarts_hyper, &state->rng);

            /* Find best observed score */
            double f_best = -1e30;
            for (int32_t i = 0; i < n_train; i++)
                if (y_train[i] > f_best) f_best = y_train[i];

            /* Optimize acquisition */
            double *x_next = (double *)malloc((size_t)n_active * sizeof(double));
            bo_optimize_acquisition(&gp, state->acq_fn,
                                    state->kappa, state->xi, f_best,
                                    state->n_restarts_acq, &state->rng,
                                    x_next);

            /* Decode to raw params */
            bo_space_decode(&state->space, x_next, cat_vals, params, n_active);
            free(x_next);
            bo_gp_free(&gp);
            free(X_train); free(y_train); free(cat_vals);
            return 0;
        }

        /* GP failed: fall through to random */
        bo_gp_free(&gp);
        free(X_train); free(y_train);
    }

    /* Random fallback: sample uniform in [0,1]^n_active, decode */
    double *x_rand = (double *)malloc((size_t)n_active * sizeof(double));
    for (int32_t d = 0; d < n_active; d++)
        x_rand[d] = bo_rng_uniform(&state->rng);
    bo_space_decode(&state->space, x_rand, cat_vals, params, n_active);
    free(x_rand);
    free(cat_vals);
    return 0;
}

int bo_suggest_liar(bo_state_t *state, double *params, double hallucinated_y) {
    /* Temporarily add a hallucinated observation, suggest, then remove it */
    double *prev_params = (double *)malloc(
        (size_t)state->space.n_dims * sizeof(double));
    if (!prev_params) { bo_set_error("alloc liar"); return -1; }

    /* First suggest to get params */
    int ret = bo_suggest(state, params);
    if (ret != 0) { free(prev_params); return ret; }

    /* Observe with hallucinated score */
    memcpy(prev_params, params, (size_t)state->space.n_dims * sizeof(double));
    bo_observe(state, prev_params, hallucinated_y);

    free(prev_params);
    return 0;
}

int32_t bo_get_n_obs(const bo_state_t *state) {
    return state->n_obs;
}

double bo_get_best_score(const bo_state_t *state) {
    double best = -1e30;
    for (int32_t i = 0; i < state->n_obs; i++)
        if (state->obs[i].score > best)
            best = state->obs[i].score;
    return best;
}

int32_t bo_get_n_contexts(const bo_state_t *state) {
    return state->n_contexts;
}

void bo_free(bo_state_t *state) {
    if (!state) return;

    /* Free observations */
    for (int32_t i = 0; i < state->n_obs; i++)
        free(state->obs[i].values);
    free(state->obs);

    /* Free contexts */
    for (int32_t i = 0; i < state->n_contexts; i++)
        context_free(&state->contexts[i]);
    free(state->contexts);

    /* Free categorical samplers */
    for (int32_t k = 0; k < state->space.n_categorical; k++)
        cat_sampler_free(&state->cat_samplers[k]);
    free(state->cat_samplers);

    /* Free space */
    free(state->space.dims);
    free(state->space.cat_ids);
    free(state->space.cont_ids);

    free(state);
}
