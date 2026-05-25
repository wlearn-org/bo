/*
 * test_c.c -- Native C tests for BO engine
 *
 * Compile: cc -std=c11 -O2 -o test_bo bo.c test_c.c -lm
 * Run: ./test_bo
 */

#define _USE_MATH_DEFINES
#include "bo.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Internal helpers from bo.c -- not static so we can test them */
extern void cat_sampler_init(bo_cat_sampler_t *cs, int32_t n_values);
extern int32_t cat_sampler_sample(bo_cat_sampler_t *cs, bo_rng_t *rng);
extern void cat_sampler_observe(bo_cat_sampler_t *cs,
                                int32_t value, double score, double median);
extern void cat_sampler_free(bo_cat_sampler_t *cs);

static int passed = 0, failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d: %s)\n", msg, __LINE__, #cond); \
        failed++; \
        return; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    double _a = (a), _b = (b); \
    if (fabs(_a - _b) > (tol)) { \
        printf("  FAIL: %s (line %d: %.8f != %.8f, tol=%.1e)\n", \
               msg, __LINE__, _a, _b, (double)(tol)); \
        failed++; \
        return; \
    } \
} while(0)

#define PASS(msg) do { printf("  PASS: %s\n", msg); passed++; } while(0)

/* ---- PRNG ---- */

static void test_prng(void) {
    bo_rng_t rng = { .state = 42 };
    /* Should match @wlearn/core LCG exactly */
    uint32_t v1 = bo_rng_next(&rng);
    ASSERT(v1 == ((42u * 1664525u + 1013904223u) & 0x7FFFFFFFu), "prng first value");
    /* Uniform in [0,1) */
    double u = bo_rng_uniform(&rng);
    ASSERT(u >= 0.0 && u <= 1.0, "prng uniform range");
    PASS("prng");
}

/* ---- Normal distribution ---- */

static void test_rng_normal(void) {
    bo_rng_t rng = { .state = 123 };
    double sum = 0.0, sum2 = 0.0;
    int N = 10000;
    for (int i = 0; i < N; i++) {
        double x = bo_rng_normal(&rng);
        sum += x;
        sum2 += x * x;
    }
    double mean = sum / N;
    double var = sum2 / N - mean * mean;
    ASSERT_NEAR(mean, 0.0, 0.05, "normal mean");
    ASSERT_NEAR(var, 1.0, 0.1, "normal variance");
    PASS("rng_normal");
}

/* ---- Beta distribution ---- */

static void test_rng_beta(void) {
    bo_rng_t rng = { .state = 456 };
    double sum = 0.0;
    int N = 10000;
    double a = 2.0, b = 5.0;
    for (int i = 0; i < N; i++)
        sum += bo_rng_beta(&rng, a, b);
    double mean = sum / N;
    double expected = a / (a + b);
    ASSERT_NEAR(mean, expected, 0.02, "beta mean");
    PASS("rng_beta");
}

/* ---- Norm CDF ---- */

static void test_norm_cdf(void) {
    ASSERT_NEAR(bo_norm_cdf(0.0), 0.5, 1e-5, "norm_cdf(0)");
    ASSERT_NEAR(bo_norm_cdf(1.96), 0.975, 1e-3, "norm_cdf(1.96)");
    ASSERT_NEAR(bo_norm_cdf(-1.96), 0.025, 1e-3, "norm_cdf(-1.96)");
    PASS("norm_cdf");
}

/* ---- Cholesky ---- */

static void test_cholesky(void) {
    /* 3x3 PD matrix (col-major):
     * [4 2 1]
     * [2 5 3]
     * [1 3 6]  */
    double A[9] = {4, 2, 1,  2, 5, 3,  1, 3, 6};
    int ret = bo_cholesky(A, 3);
    ASSERT(ret == 0, "cholesky returns 0");
    /* L[0,0] = sqrt(4) = 2 */
    ASSERT_NEAR(A[0], 2.0, 1e-10, "L[0,0]");
    /* L[1,0] = 2/2 = 1 */
    ASSERT_NEAR(A[1], 1.0, 1e-10, "L[1,0]");
    /* L[2,0] = 1/2 = 0.5 */
    ASSERT_NEAR(A[2], 0.5, 1e-10, "L[2,0]");
    /* Verify L L^T = original by checking a few entries */
    /* L[1,1] = sqrt(5 - 1) = 2 */
    ASSERT_NEAR(A[3 + 1], 2.0, 1e-10, "L[1,1]");
    PASS("cholesky");
}

/* ---- Cholesky jitter ---- */

static void test_cholesky_jitter(void) {
    /* Near-singular matrix */
    double A[4] = {1.0, 1.0,  1.0, 1.0};
    double jitter = bo_cholesky_jitter(A, 2);
    ASSERT(jitter > 0.0, "jitter > 0 for singular matrix");
    ASSERT(jitter <= 1e-3, "jitter reasonable");
    PASS("cholesky_jitter");
}

/* ---- Trisolve ---- */

static void test_trisolve(void) {
    /* L = [[2 0], [1 3]] (col-major: {2,1, 0,3}) */
    double L[4] = {2, 1,  0, 3};
    double b[2] = {4, 7};
    /* Lx = b: x = [2, 5/3] */
    bo_trisolve_lower(L, 2, b);
    ASSERT_NEAR(b[0], 2.0, 1e-10, "trisolve_lower x[0]");
    ASSERT_NEAR(b[1], 5.0/3.0, 1e-10, "trisolve_lower x[1]");
    PASS("trisolve");
}

/* ---- Kernel ---- */

static void test_kernel(void) {
    bo_hypers_t h;
    double ls[2] = {1.0, 1.0};
    h.lengthscales = ls;
    h.signal_var = 1.0;
    h.noise_var = 0.01;

    double x1[2] = {0.0, 0.0};
    double x2[2] = {0.0, 0.0};

    /* k(x, x) = signal_var */
    double k = bo_kernel_matern52(x1, x2, 2, &h);
    ASSERT_NEAR(k, 1.0, 1e-10, "kernel k(x,x) = signal_var");

    /* k at large distance -> 0 */
    double x3[2] = {100.0, 100.0};
    k = bo_kernel_matern52(x1, x3, 2, &h);
    ASSERT(k < 1e-10, "kernel k(far) ~= 0");

    PASS("kernel");
}

/* ---- GP fit and predict ---- */

static void test_gp_fit_predict(void) {
    /* 5 points of sin(x) in [0, 2pi] */
    int n = 5, dim = 1;
    double X[5] = {0.0, 0.5, 1.0, 1.5, 2.0};
    double y[5];
    for (int i = 0; i < n; i++)
        y[i] = sin(X[i] * 3.14159);

    bo_hypers_t h;
    double ls[1] = {0.5};
    h.lengthscales = ls;
    h.signal_var = 1.0;
    h.noise_var = 0.001;

    bo_gp_t gp;
    memset(&gp, 0, sizeof(gp));
    int ret = bo_gp_fit(&gp, X, y, n, dim, &h, BO_KERNEL_MATERN52);
    ASSERT(ret == 0, "gp_fit succeeds");

    /* Predict at training point: mean should be close to y */
    double mu, var;
    double x_test[1] = {0.5};
    bo_gp_predict(&gp, x_test, &mu, &var);
    ASSERT_NEAR(mu, sin(0.5 * 3.14159), 0.1, "gp predict at training point");
    ASSERT(var < 0.1, "gp variance at training point is small");

    /* Predict far away: variance should be larger */
    double x_far[1] = {3.0};
    bo_gp_predict(&gp, x_far, &mu, &var);
    ASSERT(var > 0.1, "gp variance far from training data is larger");

    bo_gp_free(&gp);
    PASS("gp_fit_predict");
}

/* ---- L-BFGS ---- */

/* Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2 */
static double rosenbrock(const double *x, double *grad, int32_t n, void *ctx) {
    (void)ctx; (void)n;
    double a = x[0], b = x[1];
    double f = (1.0 - a) * (1.0 - a) + 100.0 * (b - a * a) * (b - a * a);
    if (grad) {
        grad[0] = -2.0 * (1.0 - a) - 400.0 * a * (b - a * a);
        grad[1] = 200.0 * (b - a * a);
    }
    return f;
}

static void test_lbfgs(void) {
    double x[2] = {-1.0, 1.0};
    double lower[2] = {-5.0, -5.0};
    double upper[2] = {5.0, 5.0};
    double f = bo_lbfgs_minimize(rosenbrock, NULL, x, 2, lower, upper, 200, 1e-10);
    ASSERT_NEAR(x[0], 1.0, 0.01, "lbfgs rosenbrock x[0]");
    ASSERT_NEAR(x[1], 1.0, 0.01, "lbfgs rosenbrock x[1]");
    ASSERT(f < 0.001, "lbfgs rosenbrock f ~= 0");
    PASS("lbfgs");
}

/* ---- Acquisition functions ---- */

static void test_acquisition(void) {
    /* EI at f_best: should be ~0 */
    double ei = bo_acq_ei(0.5, 0.1, 0.5, 0.0);
    ASSERT(ei < 0.05, "ei at f_best is small");

    /* EI below f_best with positive sigma: should be > 0 */
    ei = bo_acq_ei(0.8, 0.2, 0.5, 0.0);
    ASSERT(ei > 0.0, "ei above f_best is positive");

    /* EI monotone in sigma */
    double ei1 = bo_acq_ei(0.6, 0.1, 0.5, 0.0);
    double ei2 = bo_acq_ei(0.6, 0.5, 0.5, 0.0);
    ASSERT(ei2 > ei1, "ei increases with sigma");

    /* UCB increases with kappa */
    double ucb1 = bo_acq_ucb(0.5, 0.3, 1.0);
    double ucb2 = bo_acq_ucb(0.5, 0.3, 3.0);
    ASSERT(ucb2 > ucb1, "ucb increases with kappa");

    PASS("acquisition");
}

/* ---- Search space ---- */

static void test_space(void) {
    /* 3 dims: continuous [0,10], integer [1,5], categorical (3 values) */
    bo_space_t *s = bo_space_create(3);
    ASSERT(s != NULL, "space_create");
    bo_space_add_continuous(s, 0, 0.0, 10.0, 0);
    bo_space_add_integer(s, 1, 1.0, 5.0, 0);
    bo_space_add_categorical(s, 2, 3);
    bo_space_finalize(s);

    ASSERT(s->n_continuous == 2, "2 continuous dims");
    ASSERT(s->n_categorical == 1, "1 categorical dim");

    /* Encode/decode round-trip */
    double raw[3] = {5.0, 3.0, 1.0};  /* cat value = 1 */
    int32_t cat_vals[1] = {1};
    double encoded[2];
    int32_t n_active = bo_space_encode(s, raw, cat_vals, encoded);
    ASSERT(n_active == 2, "2 active dims");
    ASSERT_NEAR(encoded[0], 0.5, 1e-10, "encode continuous 5/10 = 0.5");
    ASSERT_NEAR(encoded[1], 0.5, 1e-10, "encode integer (3-1)/(5-1) = 0.5");

    double decoded[3] = {0, 0, 0};
    bo_space_decode(s, encoded, cat_vals, decoded, n_active);
    ASSERT_NEAR(decoded[0], 5.0, 1e-10, "decode continuous");
    ASSERT_NEAR(decoded[1], 3.0, 1e-10, "decode integer");

    bo_space_free(s);
    PASS("space");
}

/* ---- Conditional params ---- */

static void test_conditional(void) {
    /* 3 dims: categorical (2 values), continuous [0,1], conditional continuous [0,1]
     * dim 2 is only active when dim 0 == 1 */
    bo_space_t *s = bo_space_create(3);
    bo_space_add_categorical(s, 0, 2);
    bo_space_add_continuous(s, 1, 0.0, 1.0, 0);
    bo_space_add_continuous(s, 2, 0.0, 1.0, 0);
    bo_space_add_condition(s, 2, 0, 1);
    bo_space_finalize(s);

    int32_t cat0[1] = {0};
    int32_t cat1[1] = {1};

    ASSERT(bo_space_count_active(s, cat0) == 1, "cat=0: 1 active");
    ASSERT(bo_space_count_active(s, cat1) == 2, "cat=1: 2 active");

    bo_space_free(s);
    PASS("conditional");
}

/* ---- Thompson sampling ---- */

static void test_thompson(void) {
    bo_cat_sampler_t cs;
    cat_sampler_init(&cs, 3);
    bo_rng_t rng = { .state = 42 };

    /* All values should be sampled eventually */
    int counts[3] = {0, 0, 0};
    for (int i = 0; i < 100; i++) {
        int32_t v = cat_sampler_sample(&cs, &rng);
        ASSERT(v >= 0 && v < 3, "thompson value in range");
        counts[v]++;
    }
    ASSERT(counts[0] > 0 && counts[1] > 0 && counts[2] > 0,
           "thompson samples all values");

    /* After observing that value 1 is always best, it should be preferred */
    for (int i = 0; i < 20; i++) {
        cat_sampler_observe(&cs, 1, 1.0, 0.5);
        cat_sampler_observe(&cs, 0, 0.0, 0.5);
        cat_sampler_observe(&cs, 2, 0.0, 0.5);
    }
    int best_count = 0;
    for (int i = 0; i < 100; i++) {
        if (cat_sampler_sample(&cs, &rng) == 1) best_count++;
    }
    ASSERT(best_count > 60, "thompson adapts to best value");

    cat_sampler_free(&cs);
    PASS("thompson");
}

/* ---- Full BO loop ---- */

static double branin(double x1, double x2) {
    /* Branin function: well-known 2D test function
     * Input domain: x1 in [-5, 10], x2 in [0, 15]
     * Global min ~= 0.397887 */
    double a = 1.0, b = 5.1 / (4.0 * M_PI * M_PI);
    double c = 5.0 / M_PI, r = 6.0, s = 10.0;
    double t = 1.0 / (8.0 * M_PI);
    double term = x2 - b * x1 * x1 + c * x1 - r;
    return a * term * term + s * (1.0 - t) * cos(x1) + s;
}

static void test_full_bo(void) {
    /* 2 continuous dims: [−5, 10] and [0, 15] */
    bo_space_t *space = bo_space_create(2);
    bo_space_add_continuous(space, 0, -5.0, 10.0, 0);
    bo_space_add_continuous(space, 1, 0.0, 15.0, 0);
    bo_space_finalize(space);

    bo_state_t *state = bo_create(space, BO_KERNEL_MATERN52, BO_ACQ_EI,
                                  2.0, 0.01, 5, 20, 500, 3, 42);
    ASSERT(state != NULL, "bo_create");

    /* Initial random observations */
    bo_rng_t init_rng = { .state = 123 };
    for (int i = 0; i < 10; i++) {
        double params[2];
        params[0] = -5.0 + bo_rng_uniform(&init_rng) * 15.0;
        params[1] = bo_rng_uniform(&init_rng) * 15.0;
        double score = -branin(params[0], params[1]);  /* negate for maximization */
        bo_observe(state, params, score);
    }

    /* BO loop */
    for (int i = 0; i < 40; i++) {
        double params[2];
        int ret = bo_suggest(state, params);
        ASSERT(ret == 0, "bo_suggest ok");
        ASSERT(params[0] >= -5.0 && params[0] <= 10.0, "param[0] in bounds");
        ASSERT(params[1] >= 0.0 && params[1] <= 15.0, "param[1] in bounds");
        double score = -branin(params[0], params[1]);
        bo_observe(state, params, score);
    }

    double best = bo_get_best_score(state);
    /* Branin global min is -0.3979. With 10 initial random + 40 BO evals,
     * we consistently reach ~-3.8. Threshold at -10 catches regressions
     * while allowing for seed/platform variance. */
    printf("    Branin best: %.4f (target: -0.3979)\n", best);
    ASSERT(best > -10.0, "bo converges on Branin (expect > -10)");

    ASSERT(bo_get_n_obs(state) == 50, "50 observations");

    bo_free(state);
    bo_space_free(space);
    PASS("full_bo");
}

/* ---- Full BO with categoricals ---- */

static void test_bo_categorical(void) {
    /* 1 categorical (3 values) + 1 continuous [0, 1]
     * Objective: cat=1 is best, f(x) = -(x-0.7)^2 for cat=1 */
    bo_space_t *space = bo_space_create(2);
    bo_space_add_categorical(space, 0, 3);
    bo_space_add_continuous(space, 1, 0.0, 1.0, 0);
    bo_space_finalize(space);

    bo_state_t *state = bo_create(space, BO_KERNEL_MATERN52, BO_ACQ_EI,
                                  2.0, 0.01, 3, 10, 500, 3, 99);
    ASSERT(state != NULL, "bo_create categorical");

    /* Initial random */
    bo_rng_t init_rng = { .state = 77 };
    for (int i = 0; i < 9; i++) {
        double params[2];
        params[0] = (double)bo_rng_int(&init_rng, 3);  /* cat */
        params[1] = bo_rng_uniform(&init_rng);           /* cont */
        int cat = (int)params[0];
        double score = (cat == 1) ? -(params[1] - 0.7) * (params[1] - 0.7)
                                  : -(params[1] - 0.3) * (params[1] - 0.3) - 0.5;
        bo_observe(state, params, score);
    }

    /* BO loop */
    for (int i = 0; i < 21; i++) {
        double params[2];
        int ret = bo_suggest(state, params);
        ASSERT(ret == 0, "bo_suggest categorical ok");
        int cat = (int)params[0];
        ASSERT(cat >= 0 && cat < 3, "cat in range");
        double score = (cat == 1) ? -(params[1] - 0.7) * (params[1] - 0.7)
                                  : -(params[1] - 0.3) * (params[1] - 0.3) - 0.5;
        bo_observe(state, params, score);
    }

    double best = bo_get_best_score(state);
    printf("    Categorical best: %.4f (target: 0.0)\n", best);
    ASSERT(best > -0.1, "bo finds good categorical solution");
    ASSERT(bo_get_n_contexts(state) >= 1, "at least 1 context");

    bo_free(state);
    bo_space_free(space);
    PASS("bo_categorical");
}

/* ---- Determinism ---- */

static void test_determinism(void) {
    /* Two identical runs should produce identical suggestions */
    bo_space_t *space = bo_space_create(1);
    bo_space_add_continuous(space, 0, 0.0, 1.0, 0);
    bo_space_finalize(space);

    double results[2][5];
    for (int run = 0; run < 2; run++) {
        bo_state_t *state = bo_create(space, BO_KERNEL_MATERN52, BO_ACQ_EI,
                                      2.0, 0.01, 3, 10, 500, 3, 42);
        /* Seed with same observations */
        double obs_x[3] = {0.1, 0.5, 0.9};
        double obs_y[3] = {0.2, 0.8, 0.3};
        for (int i = 0; i < 3; i++) {
            double params[1] = {obs_x[i]};
            bo_observe(state, params, obs_y[i]);
        }
        /* Suggest 5 times */
        for (int i = 0; i < 5; i++) {
            double params[1];
            bo_suggest(state, params);
            results[run][i] = params[0];
            bo_observe(state, params, sin(params[0]));
        }
        bo_free(state);
    }

    for (int i = 0; i < 5; i++)
        ASSERT_NEAR(results[0][i], results[1][i], 1e-10, "determinism");

    bo_space_free(space);
    PASS("determinism");
}

/* ---- Degenerate data ---- */

static void test_degenerate(void) {
    bo_space_t *space = bo_space_create(1);
    bo_space_add_continuous(space, 0, 0.0, 1.0, 0);
    bo_space_finalize(space);

    bo_state_t *state = bo_create(space, BO_KERNEL_MATERN52, BO_ACQ_EI,
                                  2.0, 0.01, 3, 10, 500, 3, 42);

    /* All same score */
    for (int i = 0; i < 5; i++) {
        double params[1] = {0.1 * (i + 1)};
        bo_observe(state, params, 0.5);
    }

    /* Should not crash */
    double params[1];
    int ret = bo_suggest(state, params);
    ASSERT(ret == 0, "degenerate suggest ok");
    ASSERT(params[0] >= 0.0 && params[0] <= 1.0, "degenerate in bounds");

    bo_free(state);
    bo_space_free(space);
    PASS("degenerate");
}

/* ---- Dispose safety ---- */

static void test_dispose(void) {
    bo_space_t *space = bo_space_create(1);
    bo_space_add_continuous(space, 0, 0.0, 1.0, 0);
    bo_space_finalize(space);

    bo_state_t *state = bo_create(space, BO_KERNEL_MATERN52, BO_ACQ_EI,
                                  2.0, 0.01, 3, 10, 500, 3, 42);
    bo_free(state);
    bo_free(NULL);  /* should not crash */
    bo_space_free(space);
    PASS("dispose");
}

/* ---- Main ---- */

int main(void) {
    printf("BO native C tests\n");
    printf("==================\n");

    test_prng();
    test_rng_normal();
    test_rng_beta();
    test_norm_cdf();
    test_cholesky();
    test_cholesky_jitter();
    test_trisolve();
    test_kernel();
    test_gp_fit_predict();
    test_lbfgs();
    test_acquisition();
    test_space();
    test_conditional();
    test_thompson();
    test_full_bo();
    test_bo_categorical();
    test_determinism();
    test_degenerate();
    test_dispose();

    printf("==================\n");
    printf("%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
