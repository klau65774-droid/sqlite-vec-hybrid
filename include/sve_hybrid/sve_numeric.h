#ifndef SVE_NUMERIC_H
#define SVE_NUMERIC_H

#include "sve_config.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t sve_crc32c(const void *data, size_t len);
uint32_t sve_crc32c_update(uint32_t crc, const void *data, size_t len);

/* Numerically stable distance kernels (overflow/underflow/zero-norm safe) */
int sve_distance_l2_f32(const float *a, const float *b, size_t dims,
                        double *out_dist);
int sve_distance_l2_i8(const int8_t *a, const int8_t *b, size_t dims,
                       double *out_dist);
int sve_distance_l1_f32(const float *a, const float *b, size_t dims,
                        double *out_dist);
int sve_distance_l1_i8(const int8_t *a, const int8_t *b, size_t dims,
                       int64_t *out_dist);
int sve_distance_cosine_f32(const float *a, const float *b, size_t dims,
                            double *out_dist);
int sve_distance_cosine_i8(const int8_t *a, const int8_t *b, size_t dims,
                           double *out_dist);
int sve_distance_ip_f32(const float *a, const float *b, size_t dims,
                        double *out_dist);
int sve_distance_hamming(const uint8_t *a, const uint8_t *b, size_t dims_bits,
                         uint64_t *out_dist);

/* Symmetric absmax SQ8 quantization & fast asymmetric/symmetric distance */
int sve_quantize_f32_to_sq8(const float *src, size_t dims, int8_t *dst_q,
                            float *out_scale);
int sve_dequantize_sq8_to_f32(const int8_t *src_q, float scale, size_t dims,
                              float *dst);
double sve_distance_sq8_l2(const int8_t *qa, float sa, const int8_t *qb,
                           float sb, size_t dims);
double sve_distance_sq8_cosine(const int8_t *qa, const int8_t *qb, size_t dims);

/* Checked / saturating element-wise vector arithmetic (#363 fix) */
int sve_add_f32(const float *a, const float *b, float *out, size_t dims);
int sve_sub_f32(const float *a, const float *b, float *out, size_t dims);
int sve_add_i8(const int8_t *a, const int8_t *b, int8_t *out, size_t dims,
               bool saturate);
int sve_sub_i8(const int8_t *a, const int8_t *b, int8_t *out, size_t dims,
               bool saturate);

/* Strictly bounded JSON vector parser */
int sve_parse_f32_json(const char *json, size_t json_len, size_t max_dims,
                       float **out_vec, size_t *out_dims);
int sve_parse_i8_json(const char *json, size_t json_len, size_t max_dims,
                      int8_t **out_vec, size_t *out_dims);

/* Dispatch scalar distance between two typed arrays */
int sve_compute_distance(sve_elem_type_t elem_type, sve_metric_t metric,
                         const void *a, const void *b, size_t dims,
                         double *out_dist);

size_t sve_elem_byte_size(sve_elem_type_t elem_type, size_t dims);
bool sve_f32_array_is_finite(const float *v, size_t dims);

#ifdef __cplusplus
}
#endif

#endif /* SVE_NUMERIC_H */
