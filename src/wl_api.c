/*
 * wl_api.c -- C ABI wrapper for BO (WASM/FFI boundary)
 *
 * All params as primitives (no struct passing across ABI).
 */

#include "bo.h"

const char *wl_bo_get_last_error(void) {
    return bo_get_error();
}

/* ---------- Space construction ---------- */

void *wl_bo_space_create(int n_dims) {
    return (void *)bo_space_create((int32_t)n_dims);
}

int wl_bo_space_add_continuous(void *sp, int id,
                               double low, double high, int log_scale) {
    return bo_space_add_continuous((bo_space_t *)sp,
                                  (int32_t)id, low, high, (int32_t)log_scale);
}

int wl_bo_space_add_integer(void *sp, int id,
                            double low, double high, int log_scale) {
    return bo_space_add_integer((bo_space_t *)sp,
                                (int32_t)id, low, high, (int32_t)log_scale);
}

int wl_bo_space_add_categorical(void *sp, int id, int n_values) {
    return bo_space_add_categorical((bo_space_t *)sp,
                                    (int32_t)id, (int32_t)n_values);
}

int wl_bo_space_add_condition(void *sp, int id,
                              int parent_id, int parent_value) {
    return bo_space_add_condition((bo_space_t *)sp,
                                  (int32_t)id, (int32_t)parent_id,
                                  (int32_t)parent_value);
}

void wl_bo_space_finalize(void *sp) {
    bo_space_finalize((bo_space_t *)sp);
}

void wl_bo_space_free(void *sp) {
    bo_space_free((bo_space_t *)sp);
}

/* ---------- Optimizer ---------- */

void *wl_bo_create(void *space, int kernel, int acq,
                   double kappa, double xi,
                   int n_restarts_hyper, int n_restarts_acq,
                   int max_obs, int min_obs_for_gp, int seed) {
    return (void *)bo_create((const bo_space_t *)space,
                             (int32_t)kernel, (int32_t)acq,
                             kappa, xi,
                             (int32_t)n_restarts_hyper, (int32_t)n_restarts_acq,
                             (int32_t)max_obs, (int32_t)min_obs_for_gp,
                             (uint32_t)seed);
}

int wl_bo_observe(void *state, const double *params, double score) {
    return bo_observe((bo_state_t *)state, params, score);
}

int wl_bo_suggest(void *state, double *params) {
    return bo_suggest((bo_state_t *)state, params);
}

int wl_bo_suggest_liar(void *state, double *params, double hallucinated_y) {
    return bo_suggest_liar((bo_state_t *)state, params, hallucinated_y);
}

/* ---------- Queries ---------- */

int wl_bo_get_n_obs(void *state) {
    return (int)bo_get_n_obs((const bo_state_t *)state);
}

double wl_bo_get_best_score(void *state) {
    return bo_get_best_score((const bo_state_t *)state);
}

int wl_bo_get_n_contexts(void *state) {
    return (int)bo_get_n_contexts((const bo_state_t *)state);
}

void wl_bo_free(void *state) {
    bo_free((bo_state_t *)state);
}
