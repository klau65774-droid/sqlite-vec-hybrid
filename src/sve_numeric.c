#include "sve_hybrid/sve_numeric.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(float) == sizeof(uint32_t) && FLT_RADIX == 2 &&
               FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
               DBL_MANT_DIG >= 53 && DBL_MAX_EXP >= 1024,
               "sve requires binary32 float and at least binary64 double");

/* Castagnoli (0x82F63B78) nibble lookup table (16 entries, thread-safe const) */
static const uint32_t k_crc32c_nibble[16] = {
    0x00000000U, 0x105EC76FU, 0x20BD8EDEU, 0x30E349B1U,
    0x417B1DBCU, 0x5125DAD3U, 0x61C69362U, 0x7198540DU,
    0x82F63B78U, 0x92A8FC17U, 0xA24BB5A6U, 0xB21572C9U,
    0xC38D26C4U, 0xD3D3E1ABU, 0xE330A81AU, 0xF36E6F75U};

uint32_t sve_crc32c_update(uint32_t crc, const void *data, size_t len) {
  if (!data && len > 0) {
    return crc;
  }
  const uint8_t *p = (const uint8_t *)data;
  uint32_t c = ~crc;
  for (size_t i = 0; i < len; i++) {
    uint8_t b = p[i];
    c = (c >> 4U) ^ k_crc32c_nibble[(c ^ b) & 0x0FU];
    c = (c >> 4U) ^ k_crc32c_nibble[(c ^ (b >> 4U)) & 0x0FU];
  }
  return ~c;
}

uint32_t sve_crc32c(const void *data, size_t len) {
  return sve_crc32c_update(0U, data, len);
}

size_t sve_elem_byte_size(sve_elem_type_t elem_type, size_t dims) {
  switch (elem_type) {
  case SVE_ELEM_FLOAT32:
    return dims <= SIZE_MAX / sizeof(float) ? dims * sizeof(float) : 0U;
  case SVE_ELEM_INT8:
    return dims * sizeof(int8_t);
  case SVE_ELEM_BIT:
    return dims / 8U + (dims % 8U != 0U);
  default:
    return 0U;
  }
}

bool sve_f32_array_is_finite(const float *v, size_t dims) {
  if (!v || dims == 0U) {
    return false;
  }
  for (size_t i = 0; i < dims; i++) {
    if (!isfinite(v[i])) {
      return false;
    }
  }
  return true;
}

int sve_distance_l2_f32(const float *a, const float *b, size_t dims,
                        double *out_dist) {
  if (!a || !b || !out_dist || dims == 0U || dims > SVE_MAX_SCALAR_DIMS) {
    return SVE_ERR_INVALID;
  }
  /*
   * Every finite float32 difference squared lies in [0, 4.64e77] and above
   * subnormal threshold (> 1e-90), so accumulating in IEEE-754 double cannot
   * overflow or underflow to zero like upstream float32 accumulators (#334).
   */
  double sum = 0.0;
  for (size_t i = 0; i < dims; i++) {
    if (!isfinite(a[i]) || !isfinite(b[i])) {
      return SVE_ERR_INVALID;
    }
    double d = (double)a[i] - (double)b[i];
    sum += d * d;
  }
  if (!isfinite(sum)) {
    return SVE_ERR_OVERFLOW;
  }
  *out_dist = sqrt(sum);
  return SVE_OK;
}

int sve_distance_l2_i8(const int8_t *a, const int8_t *b, size_t dims,
                       double *out_dist) {
  if (!a || !b || !out_dist || dims == 0U || dims > SVE_MAX_SCALAR_DIMS) {
    return SVE_ERR_INVALID;
  }
  double sum = 0.0;
  for (size_t i = 0; i < dims; i++) {
    int32_t d = (int32_t)a[i] - (int32_t)b[i];
    sum += (double)d * (double)d;
  }
  *out_dist = sqrt(sum);
  return SVE_OK;
}

int sve_distance_l1_f32(const float *a, const float *b, size_t dims,
                        double *out_dist) {
  if (!a || !b || !out_dist || dims == 0U || dims > SVE_MAX_SCALAR_DIMS) {
    return SVE_ERR_INVALID;
  }
  double sum = 0.0;
  for (size_t i = 0; i < dims; i++) {
    if (!isfinite(a[i]) || !isfinite(b[i])) {
      return SVE_ERR_INVALID;
    }
    sum += fabs((double)a[i] - (double)b[i]);
  }
  if (!isfinite(sum)) {
    return SVE_ERR_OVERFLOW;
  }
  *out_dist = sum;
  return SVE_OK;
}

int sve_distance_l1_i8(const int8_t *a, const int8_t *b, size_t dims,
                       int64_t *out_dist) {
  if (!a || !b || !out_dist || dims == 0U || dims > SVE_MAX_SCALAR_DIMS) {
    return SVE_ERR_INVALID;
  }
  int64_t sum = 0;
  for (size_t i = 0; i < dims; i++) {
    int32_t d = (int32_t)a[i] - (int32_t)b[i];
    sum += (int64_t)(d < 0 ? -d : d);
  }
  *out_dist = sum;
  return SVE_OK;
}

int sve_distance_cosine_f32(const float *a, const float *b, size_t dims,
                            double *out_dist) {
  if (!a || !b || !out_dist || dims == 0U || dims > SVE_MAX_SCALAR_DIMS) {
    return SVE_ERR_INVALID;
  }
  double dot = 0.0;
  double norm_a = 0.0;
  double norm_b = 0.0;
  for (size_t i = 0; i < dims; i++) {
    if (!isfinite(a[i]) || !isfinite(b[i])) {
      return SVE_ERR_INVALID;
    }
    double va = (double)a[i];
    double vb = (double)b[i];
    dot += va * vb;
    norm_a += va * va;
    norm_b += vb * vb;
  }
  if (!isfinite(dot) || !isfinite(norm_a) || !isfinite(norm_b)) {
    return SVE_ERR_OVERFLOW;
  }
  if (norm_a <= 0.0 || norm_b <= 0.0) {
    *out_dist = 1.0;
    return SVE_OK;
  }
  double denom = sqrt(norm_a) * sqrt(norm_b);
  if (denom <= 0.0 || !isfinite(denom)) {
    *out_dist = 1.0;
    return SVE_OK;
  }
  double sim = dot / denom;
  if (sim > 1.0) {
    sim = 1.0;
  } else if (sim < -1.0) {
    sim = -1.0;
  }
  *out_dist = 1.0 - sim;
  return SVE_OK;
}

int sve_distance_cosine_i8(const int8_t *a, const int8_t *b, size_t dims,
                           double *out_dist) {
  if (!a || !b || !out_dist || dims == 0U || dims > SVE_MAX_SCALAR_DIMS) {
    return SVE_ERR_INVALID;
  }
  double dot = 0.0;
  double norm_a = 0.0;
  double norm_b = 0.0;
  for (size_t i = 0; i < dims; i++) {
    double va = (double)a[i];
    double vb = (double)b[i];
    dot += va * vb;
    norm_a += va * va;
    norm_b += vb * vb;
  }
  if (norm_a <= 0.0 || norm_b <= 0.0) {
    *out_dist = 1.0;
    return SVE_OK;
  }
  double sim = dot / (sqrt(norm_a) * sqrt(norm_b));
  if (sim > 1.0) {
    sim = 1.0;
  } else if (sim < -1.0) {
    sim = -1.0;
  }
  *out_dist = 1.0 - sim;
  return SVE_OK;
}

int sve_distance_ip_f32(const float *a, const float *b, size_t dims,
                        double *out_dist) {
  if (!a || !b || !out_dist || dims == 0U || dims > SVE_MAX_SCALAR_DIMS) {
    return SVE_ERR_INVALID;
  }
  double dot = 0.0;
  for (size_t i = 0; i < dims; i++) {
    if (!isfinite(a[i]) || !isfinite(b[i])) {
      return SVE_ERR_INVALID;
    }
    dot += (double)a[i] * (double)b[i];
  }
  if (!isfinite(dot)) {
    return SVE_ERR_OVERFLOW;
  }
  *out_dist = -dot;
  return SVE_OK;
}

static unsigned int popcount_u8(uint8_t x) {
  static const unsigned int k_pop4[16] = {0, 1, 1, 2, 1, 2, 2, 3,
                                          1, 2, 2, 3, 2, 3, 3, 4};
  return k_pop4[x & 0x0FU] + k_pop4[(x >> 4U) & 0x0FU];
}

int sve_distance_hamming(const uint8_t *a, const uint8_t *b, size_t dims_bits,
                         uint64_t *out_dist) {
  if (!a || !b || !out_dist || dims_bits == 0U || (dims_bits % 8U) != 0U ||
      dims_bits > SVE_MAX_SCALAR_DIMS) {
    return SVE_ERR_INVALID;
  }
  size_t nbytes = dims_bits / 8U;
  uint64_t sum = 0U;
  for (size_t i = 0; i < nbytes; i++) {
    sum += (uint64_t)popcount_u8((uint8_t)(a[i] ^ b[i]));
  }
  *out_dist = sum;
  return SVE_OK;
}

int sve_quantize_f32_to_sq8(const float *src, size_t dims, int8_t *dst_q,
                            float *out_scale) {
  if (!src || !dst_q || !out_scale || dims == 0U ||
      dims > SVE_MAX_SCALAR_DIMS) {
    return SVE_ERR_INVALID;
  }
  double max_abs = 0.0;
  for (size_t i = 0; i < dims; i++) {
    if (!isfinite(src[i])) {
      return SVE_ERR_INVALID;
    }
    double av = fabs((double)src[i]);
    if (av > max_abs) {
      max_abs = av;
    }
  }
  if (max_abs <= 0.0) {
    memset(dst_q, 0, dims);
    *out_scale = 0.0f;
    return SVE_OK;
  }
  double scale = max_abs / 127.0;
  float stored_scale = (float)scale;
  if (stored_scale == 0.0f) {
    stored_scale = nextafterf(0.0f, 1.0f);
  }
  if ((double)stored_scale * 127.0 > (double)FLT_MAX) {
    stored_scale = nextafterf(stored_scale, 0.0f);
  }
  if (scale < (double)FLT_MIN) {
    scale = (double)stored_scale;
  }
  double inv_scale = 1.0 / scale;
  for (size_t i = 0; i < dims; i++) {
    double q = nearbyint((double)src[i] * inv_scale);
    if (q > 127.0) {
      q = 127.0;
    } else if (q < -127.0) {
      q = -127.0;
    }
    dst_q[i] = (int8_t)q;
  }
  *out_scale = stored_scale;
  return SVE_OK;
}

int sve_dequantize_sq8_to_f32(const int8_t *src_q, float scale, size_t dims,
                              float *dst) {
  if (!src_q || !dst || dims == 0U || dims > SVE_MAX_SCALAR_DIMS ||
      !isfinite(scale) || scale < 0.0f) {
    return SVE_ERR_INVALID;
  }
  for (size_t i = 0; i < dims; i++) {
    double value = (double)src_q[i] * (double)scale;
    if (fabs(value) > (double)FLT_MAX) {
      return SVE_ERR_OVERFLOW;
    }
    dst[i] = (float)value;
  }
  return SVE_OK;
}

double sve_distance_sq8_l2(const int8_t *qa, float sa, const int8_t *qb,
                           float sb, size_t dims) {
  double sum = 0.0;
  for (size_t i = 0; i < dims; i++) {
    double d = ((double)qa[i] * (double)sa) - ((double)qb[i] * (double)sb);
    sum += d * d;
  }
  return sqrt(sum);
}

double sve_distance_sq8_cosine(const int8_t *qa, const int8_t *qb,
                               size_t dims) {
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  for (size_t i = 0; i < dims; i++) {
    double a = (double)qa[i];
    double b = (double)qb[i];
    dot += a * b;
    na += a * a;
    nb += b * b;
  }
  if (na <= 0.0 || nb <= 0.0) {
    return 1.0;
  }
  double sim = dot / (sqrt(na) * sqrt(nb));
  if (sim > 1.0) {
    sim = 1.0;
  } else if (sim < -1.0) {
    sim = -1.0;
  }
  return 1.0 - sim;
}

int sve_add_f32(const float *a, const float *b, float *out, size_t dims) {
  if (!a || !b || !out || dims == 0U || dims > SVE_MAX_SCALAR_DIMS) {
    return SVE_ERR_INVALID;
  }
  for (size_t i = 0; i < dims; i++) {
    if (!isfinite(a[i]) || !isfinite(b[i])) {
      return SVE_ERR_INVALID;
    }
    double value = (double)a[i] + (double)b[i];
    if (fabs(value) > (double)FLT_MAX) {
      return SVE_ERR_OVERFLOW;
    }
    out[i] = (float)value;
  }
  return SVE_OK;
}

int sve_sub_f32(const float *a, const float *b, float *out, size_t dims) {
  if (!a || !b || !out || dims == 0U || dims > SVE_MAX_SCALAR_DIMS) {
    return SVE_ERR_INVALID;
  }
  for (size_t i = 0; i < dims; i++) {
    if (!isfinite(a[i]) || !isfinite(b[i])) {
      return SVE_ERR_INVALID;
    }
    double value = (double)a[i] - (double)b[i];
    if (fabs(value) > (double)FLT_MAX) {
      return SVE_ERR_OVERFLOW;
    }
    out[i] = (float)value;
  }
  return SVE_OK;
}

int sve_add_i8(const int8_t *a, const int8_t *b, int8_t *out, size_t dims,
               bool saturate) {
  if (!a || !b || !out || dims == 0U || dims > SVE_MAX_SCALAR_DIMS) {
    return SVE_ERR_INVALID;
  }
  for (size_t i = 0; i < dims; i++) {
    int32_t v = (int32_t)a[i] + (int32_t)b[i];
    if (v > 127 || v < -128) {
      if (!saturate) {
        return SVE_ERR_OVERFLOW;
      }
      v = (v > 127) ? 127 : -128;
    }
    out[i] = (int8_t)v;
  }
  return SVE_OK;
}

int sve_sub_i8(const int8_t *a, const int8_t *b, int8_t *out, size_t dims,
               bool saturate) {
  if (!a || !b || !out || dims == 0U || dims > SVE_MAX_SCALAR_DIMS) {
    return SVE_ERR_INVALID;
  }
  for (size_t i = 0; i < dims; i++) {
    int32_t v = (int32_t)a[i] - (int32_t)b[i];
    if (v > 127 || v < -128) {
      if (!saturate) {
        return SVE_ERR_OVERFLOW;
      }
      v = (v > 127) ? 127 : -128;
    }
    out[i] = (int8_t)v;
  }
  return SVE_OK;
}

static size_t skip_ws(const char *s, size_t len, size_t pos) {
  while (pos < len &&
         (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\r' || s[pos] == '\n')) {
    pos++;
  }
  return pos;
}

static bool decimal_digit(char c) {
  return c >= '0' && c <= '9';
}

static size_t skip_digits(const char *s, size_t len, size_t pos) {
  while (pos < len && decimal_digit(s[pos])) {
    pos++;
  }
  return pos;
}

static int parse_f32_number(const char *s, size_t len, size_t start,
                            size_t whole_end, size_t *end, float *out) {
  size_t frac_start = whole_end;
  size_t frac_end = whole_end;
  if (whole_end < len && s[whole_end] == '.') {
    frac_start++;
    frac_end = skip_digits(s, len, frac_start);
    if (frac_end == frac_start) {
      return SVE_ERR_INVALID;
    }
  }
  size_t pos = frac_end;
  int exponent = 0;
  if (pos < len && (s[pos] == 'e' || s[pos] == 'E')) {
    pos++;
    bool negative = pos < len && s[pos] == '-';
    if (pos < len && (s[pos] == '+' || s[pos] == '-')) {
      pos++;
    }
    size_t digits = pos;
    /* Beyond this bound, the significand cannot bring the value back in range. */
    const int limit = (int)SVE_MAX_JSON_BYTES + 100;
    while (pos < len && decimal_digit(s[pos])) {
      if (exponent < limit) {
        exponent = exponent * 10 + (s[pos] - '0');
      }
      pos++;
    }
    if (digits == pos) {
      return SVE_ERR_INVALID;
    }
    if (negative) {
      exponent = -exponent;
    }
  }

  /* Move the radix entirely into the exponent before calling the C runtime.
     This spelling uses no locale-dependent decimal separator. */
  size_t whole_len = whole_end - start;
  size_t frac_len = frac_end - frac_start;
  size_t n = whole_len + frac_len;
  char local[96];
  size_t cap = n + 24U;
  char *number = cap <= sizeof(local) ? local : (char *)malloc(cap);
  if (!number) {
    return SVE_ERR_NOMEM;
  }
  memcpy(number, s + start, whole_len);
  memcpy(number + whole_len, s + frac_start, frac_len);
  (void)snprintf(number + n, cap - n, "e%d", exponent - (int)frac_len);
  errno = 0;
  char *tail = NULL;
  float value = strtof(number, &tail);
  bool valid = *tail == '\0' && isfinite(value) &&
               !(errno == ERANGE && fabsf(value) == FLT_MAX);
  if (number != local) {
    free(number);
  }
  if (!valid) {
    return SVE_ERR_INVALID;
  }
  *out = value;
  *end = pos;
  return SVE_OK;
}

static int parse_json_vector(const char *json, size_t len, size_t max_dims,
                             bool floating, void **out, size_t *out_dims) {
  *out = NULL;
  *out_dims = 0U;
  if (!json || max_dims == 0U || len > SVE_MAX_JSON_BYTES) {
    return SVE_ERR_LIMIT;
  }
  size_t pos = skip_ws(json, len, 0U);
  if (pos >= len || json[pos] != '[') {
    return SVE_ERR_INVALID;
  }
  pos++;
  size_t elem_size = floating ? sizeof(float) : sizeof(int8_t);
  size_t cap = max_dims < 16U ? max_dims : 16U;
  void *buf = malloc(cap * elem_size);
  if (!buf) {
    return SVE_ERR_NOMEM;
  }
  size_t count = 0U;
  int rc = SVE_ERR_INVALID;
  for (;;) {
    size_t start = skip_ws(json, len, pos);
    bool negative = start < len && json[start] == '-';
    size_t digits = start + (negative ? 1U : 0U);
    pos = skip_digits(json, len, digits);
    if (pos == digits || (json[digits] == '0' && pos - digits > 1U)) {
      break;
    }
    if (count == max_dims) {
      rc = SVE_ERR_LIMIT;
      break;
    }
    if (count == cap) {
      size_t next = cap > max_dims / 2U ? max_dims : cap * 2U;
      void *tmp = realloc(buf, next * elem_size);
      if (!tmp) {
        rc = SVE_ERR_NOMEM;
        break;
      }
      cap = next;
      buf = tmp;
    }
    if (floating) {
      rc = parse_f32_number(json, len, start, pos, &pos, &((float *)buf)[count]);
      if (rc != SVE_OK) {
        break;
      }
      rc = SVE_ERR_INVALID;
    } else {
      int value = 0;
      for (size_t i = digits; i < pos && value <= 128; i++) {
        value = 10 * value + (json[i] - '0');
      }
      if (value > (negative ? 128 : 127)) {
        break;
      }
      ((int8_t *)buf)[count] = (int8_t)(negative ? -value : value);
    }
    count++;

    pos = skip_ws(json, len, pos);
    if (pos == len) {
      break;
    }
    if (json[pos] == ',') {
      pos++;
    } else if (json[pos] == ']' && skip_ws(json, len, pos + 1U) == len) {
      *out = buf;
      *out_dims = count;
      return SVE_OK;
    } else {
      break;
    }
  }
  free(buf);
  return rc;
}

int sve_parse_f32_json(const char *json, size_t json_len, size_t max_dims,
                       float **out_vec, size_t *out_dims) {
  if (!out_vec || !out_dims) {
    return SVE_ERR_LIMIT;
  }
  void *vec = NULL;
  int rc = parse_json_vector(json, json_len, max_dims, true, &vec, out_dims);
  *out_vec = (float *)vec;
  return rc;
}

int sve_parse_i8_json(const char *json, size_t json_len, size_t max_dims,
                      int8_t **out_vec, size_t *out_dims) {
  if (!out_vec || !out_dims) {
    return SVE_ERR_LIMIT;
  }
  void *vec = NULL;
  int rc = parse_json_vector(json, json_len, max_dims, false, &vec, out_dims);
  *out_vec = (int8_t *)vec;
  return rc;
}

int sve_compute_distance(sve_elem_type_t elem_type, sve_metric_t metric,
                         const void *a, const void *b, size_t dims,
                         double *out_dist) {
  if (!a || !b || !out_dist || dims == 0U) {
    return SVE_ERR_INVALID;
  }
  switch (elem_type) {
  case SVE_ELEM_FLOAT32: {
    const float *fa = (const float *)a;
    const float *fb = (const float *)b;
    switch (metric) {
    case SVE_METRIC_L2:
      return sve_distance_l2_f32(fa, fb, dims, out_dist);
    case SVE_METRIC_COSINE:
      return sve_distance_cosine_f32(fa, fb, dims, out_dist);
    case SVE_METRIC_L1:
      return sve_distance_l1_f32(fa, fb, dims, out_dist);
    case SVE_METRIC_IP:
      return sve_distance_ip_f32(fa, fb, dims, out_dist);
    default:
      return SVE_ERR_INVALID;
    }
  }
  case SVE_ELEM_INT8: {
    const int8_t *ia = (const int8_t *)a;
    const int8_t *ib = (const int8_t *)b;
    switch (metric) {
    case SVE_METRIC_L2:
      return sve_distance_l2_i8(ia, ib, dims, out_dist);
    case SVE_METRIC_COSINE:
      return sve_distance_cosine_i8(ia, ib, dims, out_dist);
    case SVE_METRIC_L1: {
      int64_t l1 = 0;
      int rc = sve_distance_l1_i8(ia, ib, dims, &l1);
      if (rc == SVE_OK) {
        *out_dist = (double)l1;
      }
      return rc;
    }
    default:
      return SVE_ERR_INVALID;
    }
  }
  case SVE_ELEM_BIT:
    if (metric == SVE_METRIC_HAMMING || metric == SVE_METRIC_L2) {
      uint64_t h = 0U;
      int rc =
          sve_distance_hamming((const uint8_t *)a, (const uint8_t *)b, dims, &h);
      if (rc == SVE_OK) {
        *out_dist = (double)h;
      }
      return rc;
    }
    return SVE_ERR_INVALID;
  default:
    return SVE_ERR_INVALID;
  }
}
