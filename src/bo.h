/*
 * bo.h -- Bayesian Optimization with Gaussian Processes (C11)
 *
 * GP-based surrogate optimization for mixed search spaces:
 * - ARD Matern 5/2, Matern 3/2, Squared Exponential kernels
 * - Cholesky-based GP with adaptive jitter
 * - L-BFGS for hyperparameter and acquisition optimization
 * - EI, UCB, PI acquisition functions
 * - Thompson sampling for categorical parameters
 * - Per-context GPs for categorical combinations
 * - Observation reservoir for memory-bounded operation
 */

#ifndef BO_H
#define BO_H

#include <stdint.h>
#include <stddef.h>

/* ---------- PRNG (LCG matching @wlearn/core) ---------- */

typedef struct {
    uint32_t state;
} bo_rng_t;

static inline uint32_t bo_rng_next(bo_rng_t *rng) {
    rng->state = (rng->state * 1664525u + 1013904223u) & 0x7FFFFFFFu;
    return rng->state;
}

static inline double bo_rng_uniform(bo_rng_t *rng) {
    return (double)bo_rng_next(rng) / (double)0x7FFFFFFFu;
}

static inline int32_t bo_rng_int(bo_rng_t *rng, int32_t n) {
    return (int32_t)(bo_rng_uniform(rng) * n);
}

/* Gamma(shape, 1) via Marsaglia-Tsang (shape >= 1; for shape < 1 use boost) */
double bo_rng_gamma(bo_rng_t *rng, double shape);

/* Beta(a, b) via two Gamma samples */
double bo_rng_beta(bo_rng_t *rng, double a, double b);

/* Standard normal via Box-Muller */
double bo_rng_normal(bo_rng_t *rng);

/* ---------- Enums ---------- */

enum {
    BO_KERNEL_MATERN52 = 0,
    BO_KERNEL_MATERN32 = 1,
    BO_KERNEL_SE       = 2,
};

enum {
    BO_ACQ_EI  = 0,   /* Expected Improvement */
    BO_ACQ_UCB = 1,   /* Upper Confidence Bound */
    BO_ACQ_PI  = 2,   /* Probability of Improvement */
};

enum {
    BO_DIM_CONTINUOUS  = 0,
    BO_DIM_INTEGER     = 1,
    BO_DIM_CATEGORICAL = 2,
};

/* ---------- Search space ---------- */

typedef struct {
    int32_t id;            /* param ID (0-based) */
    int32_t type;          /* BO_DIM_* */
    double  low;           /* original low bound (continuous/integer) */
    double  high;          /* original high bound (continuous/integer) */
    int32_t log_scale;     /* 1 = log-transform before normalizing */
    int32_t n_values;      /* categorical: number of values; 0 otherwise */
    int32_t cond_parent;   /* -1 if unconditional */
    int32_t cond_value;    /* required parent categorical value index */
} bo_dim_desc_t;

typedef struct {
    bo_dim_desc_t *dims;
    int32_t n_dims;         /* total param count */
    int32_t n_continuous;   /* continuous + integer dims (max possible) */
    int32_t n_categorical;  /* categorical dims */
    int32_t *cat_ids;       /* [n_categorical] indices into dims[] */
    int32_t *cont_ids;      /* [n_continuous] indices into dims[] */
    int32_t finalized;
} bo_space_t;

/* ---------- GP hyperparameters ---------- */

typedef struct {
    double *lengthscales;   /* [dim] ARD lengthscales */
    double  signal_var;     /* signal variance */
    double  noise_var;      /* observation noise variance */
} bo_hypers_t;

/* ---------- GP model ---------- */

typedef struct {
    double  *X;             /* [n * dim] row-major training inputs in [0,1] */
    double  *y;             /* [n] standardized training targets */
    int32_t  n;             /* number of training points */
    int32_t  dim;           /* input dimensionality */
    double  *L;             /* [n * n] Cholesky factor, col-major */
    double  *alpha;         /* [n] L^T \ (L \ y) */
    double   y_mean;        /* target mean (for unstandardization) */
    double   y_std;         /* target std (for unstandardization) */
    bo_hypers_t hypers;
    int32_t  kernel;        /* BO_KERNEL_* */
    double   lml;           /* cached log marginal likelihood */
} bo_gp_t;

/* ---------- Thompson sampling for one categorical param ---------- */

typedef struct {
    int32_t  n_values;
    double  *alpha;         /* [n_values] Beta alpha (wins + prior) */
    double  *beta;          /* [n_values] Beta beta (losses + prior) */
} bo_cat_sampler_t;

/* ---------- Observation record ---------- */

typedef struct {
    double  *values;        /* [n_dims] all param values as doubles:
                               continuous/integer = raw value,
                               categorical = value index (0..K-1) */
    double   score;
} bo_obs_t;

/* ---------- Per-context GP ---------- */

typedef struct {
    int32_t *cat_key;       /* [n_categorical] categorical values for this context */
    int32_t  n_obs;         /* observations matching this context */
    int32_t *obs_indices;   /* [n_obs] indices into state->obs[] */
    int32_t  obs_cap;
    int32_t  gp_fitted;     /* 1 if GP is currently valid */
    bo_gp_t  gp;
} bo_context_t;

/* ---------- BO optimizer state ---------- */

typedef struct {
    bo_space_t       space;
    bo_rng_t         rng;

    /* Configuration */
    int32_t          kernel;
    int32_t          acq_fn;
    double           kappa;            /* UCB exploration weight */
    double           xi;               /* EI/PI exploration jitter */
    int32_t          n_restarts_hyper;  /* L-BFGS restarts for hypers */
    int32_t          n_restarts_acq;    /* L-BFGS restarts for acquisition */
    int32_t          max_obs;           /* observation reservoir cap */
    int32_t          min_obs_for_gp;    /* min observations per context for GP */

    /* Categorical Thompson sampling */
    bo_cat_sampler_t *cat_samplers;    /* [n_categorical] */

    /* Observations */
    bo_obs_t        *obs;
    int32_t          n_obs;
    int32_t          obs_cap;

    /* Per-context GPs */
    bo_context_t    *contexts;
    int32_t          n_contexts;
    int32_t          ctx_cap;

    /* Running statistics for Thompson sampling */
    double           score_sum;
    double           score_count;
} bo_state_t;

/* ---------- Error handling ---------- */

const char *bo_get_error(void);

/* ---------- Linear algebra ---------- */

/* In-place Cholesky (col-major lower triangular). Returns 0 ok, -1 not PD. */
int bo_cholesky(double *A, int32_t n);

/* Cholesky with adaptive jitter. Returns jitter used, -1.0 on failure. */
double bo_cholesky_jitter(double *A, int32_t n);

/* Forward solve: L x = b (L lower triangular, col-major). b overwritten. */
void bo_trisolve_lower(const double *L, int32_t n, double *b);

/* Backward solve: L^T x = b. b overwritten. */
void bo_trisolve_upper(const double *L, int32_t n, double *b);

/* Log-determinant from Cholesky: 2 * sum(log(diag(L))) */
double bo_logdet_chol(const double *L, int32_t n);

/* ---------- Special math ---------- */

double bo_norm_pdf(double x);
double bo_norm_cdf(double x);

/* ---------- Kernels ---------- */

double bo_kernel_matern52(const double *x1, const double *x2,
                          int32_t dim, const bo_hypers_t *h);
double bo_kernel_matern32(const double *x1, const double *x2,
                          int32_t dim, const bo_hypers_t *h);
double bo_kernel_se(const double *x1, const double *x2,
                    int32_t dim, const bo_hypers_t *h);

/* Compute full kernel matrix K (col-major, n*n) */
void bo_kernel_matrix(const double *X, int32_t n, int32_t dim,
                      const bo_hypers_t *h, int32_t kernel_type,
                      double *K);

/* Compute kernel vector k* between one test point and training set */
void bo_kernel_vector(const double *x_star, const double *X_train,
                      int32_t n, int32_t dim,
                      const bo_hypers_t *h, int32_t kernel_type,
                      double *k_vec);

/* Kernel gradient: dK/d(log-lengthscale[p]), dK/d(log-signal), dK/d(log-noise)
 * dK_dl: [n_hypers * n * n] col-major gradient matrices
 * Total n_hypers = dim + 2 (dim lengthscales + signal_var + noise_var) */
void bo_kernel_matrix_grad(const double *X, int32_t n, int32_t dim,
                           const bo_hypers_t *h, int32_t kernel_type,
                           double *dK);

/* ---------- GP ---------- */

/* Fit GP. X: [n*dim] row-major in [0,1]. y: [n] raw scores (standardized internally).
 * Returns 0 ok, -1 error. Allocates gp->L, gp->alpha, gp->X, gp->y. */
int bo_gp_fit(bo_gp_t *gp, const double *X, const double *y,
              int32_t n, int32_t dim,
              const bo_hypers_t *hypers, int32_t kernel_type);

/* Posterior mean and variance at x*. Returns unstandardized values. */
void bo_gp_predict(const bo_gp_t *gp, const double *x_star,
                   double *mean, double *var);

/* Log marginal likelihood */
double bo_gp_lml(const bo_gp_t *gp);

/* LML gradient w.r.t. log-hyperparams. grad: [dim + 2] */
void bo_gp_lml_grad(const bo_gp_t *gp, const double *X,
                    double *grad);

/* Free GP internals (does not free gp itself) */
void bo_gp_free(bo_gp_t *gp);

/* ---------- L-BFGS ---------- */

/* Objective function: computes f(x) and grad_f(x). Returns f(x). */
typedef double (*bo_obj_fn)(const double *x, double *grad,
                            int32_t n, void *ctx);

/* Bounded L-BFGS minimization. x: initial point, overwritten with solution.
 * lower/upper: bounds (NULL for unbounded). Returns final objective. */
double bo_lbfgs_minimize(bo_obj_fn fn, void *ctx,
                         double *x, int32_t n,
                         const double *lower, const double *upper,
                         int32_t max_iter, double ftol);

/* ---------- Acquisition functions ---------- */

double bo_acq_ei(double mu, double sigma, double f_best, double xi);
double bo_acq_ucb(double mu, double sigma, double kappa);
double bo_acq_pi(double mu, double sigma, double f_best, double xi);

/* Evaluate acquisition and gradient w.r.t. x.
 * Returns negative acquisition (for minimization). grad: [dim] or NULL. */
double bo_acq_eval_grad(const bo_gp_t *gp, const double *x,
                        int32_t acq_fn, double kappa, double xi,
                        double f_best, double *grad);

/* Multi-start acquisition optimization in [0,1]^dim.
 * x_best: [dim] output. Returns best acquisition value. */
double bo_optimize_acquisition(const bo_gp_t *gp,
                               int32_t acq_fn, double kappa, double xi,
                               double f_best,
                               int32_t n_restarts, bo_rng_t *rng,
                               double *x_best);

/* ---------- Hyperparameter optimization ---------- */

/* Multi-start minimize(-LML). Updates gp->hypers in place.
 * X: [n*dim] training data. Returns best LML. */
double bo_optimize_hypers(bo_gp_t *gp, const double *X, const double *y,
                          int32_t n, int32_t dim,
                          int32_t n_restarts, bo_rng_t *rng);

/* ---------- Search space ---------- */

bo_space_t *bo_space_create(int32_t n_dims);
int bo_space_add_continuous(bo_space_t *s, int32_t id,
                            double low, double high, int32_t log_scale);
int bo_space_add_integer(bo_space_t *s, int32_t id,
                         double low, double high, int32_t log_scale);
int bo_space_add_categorical(bo_space_t *s, int32_t id, int32_t n_values);
int bo_space_add_condition(bo_space_t *s, int32_t id,
                           int32_t parent_id, int32_t parent_value);
void bo_space_finalize(bo_space_t *s);
void bo_space_free(bo_space_t *s);

/* Encode continuous/integer params to [0,1]. Only encodes active dims
 * given current categorical values. Returns number of active continuous dims. */
int32_t bo_space_encode(const bo_space_t *s, const double *raw,
                        const int32_t *cat_values, double *encoded);

/* Decode [0,1] back to raw continuous/integer values. */
void bo_space_decode(const bo_space_t *s, const double *encoded,
                     const int32_t *cat_values, double *raw,
                     int32_t n_active);

/* Check if dim is active given categorical context */
int bo_space_is_active(const bo_space_t *s, int32_t dim_id,
                       const int32_t *cat_values);

/* Count active continuous dims for a categorical context */
int32_t bo_space_count_active(const bo_space_t *s, const int32_t *cat_values);

/* ---------- Top-level BO ---------- */

bo_state_t *bo_create(const bo_space_t *space,
                      int32_t kernel, int32_t acq_fn,
                      double kappa, double xi,
                      int32_t n_restarts_hyper, int32_t n_restarts_acq,
                      int32_t max_obs, int32_t min_obs_for_gp,
                      uint32_t seed);

/* Observe: params is [n_dims] doubles (continuous=raw, categorical=index) */
int bo_observe(bo_state_t *state, const double *params, double score);

/* Suggest: params is [n_dims] output (continuous=raw, categorical=index) */
int bo_suggest(bo_state_t *state, double *params);

/* Suggest with constant liar for batch BO */
int bo_suggest_liar(bo_state_t *state, double *params, double hallucinated_y);

/* Query state */
int32_t bo_get_n_obs(const bo_state_t *state);
double bo_get_best_score(const bo_state_t *state);
int32_t bo_get_n_contexts(const bo_state_t *state);

void bo_free(bo_state_t *state);

#endif /* BO_H */
