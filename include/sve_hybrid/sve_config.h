#ifndef SVE_CONFIG_H
#define SVE_CONFIG_H

#include <stddef.h>
#include <stdint.h>

#define SVE_HYBRID_VERSION "v1.0.0-sve-hybrid"
#define SVE_UPSTREAM_COMMIT "04d28bd21773981e2d266bbf6aa4efbd011eb4f6"

#define SVE_MAX_TABLE_DIMS 8192U
#define SVE_MAX_SCALAR_DIMS 16777216U
#define SVE_MAX_K 4096U
#define SVE_MAX_JSON_BYTES (1024U * 1024U)
#define SVE_MAX_TEXT_BYTES (256U * 1024U)
#define SVE_MAX_USER_COLS 16U
#define SVE_MAX_META_VAL_BYTES 4096U

#define SVE_MAX_HNSW_M 64U
#define SVE_MAX_HNSW_EF 2048U
#define SVE_DEFAULT_HNSW_M 16U
#define SVE_DEFAULT_HNSW_EF_CONSTRUCTION 100U
#define SVE_DEFAULT_HNSW_EF_SEARCH 64U
#define SVE_EXACT_FILTER_THRESHOLD 256U

typedef enum sve_status {
  SVE_OK = 0,
  SVE_ERR_INVALID = -1,
  SVE_ERR_NOMEM = -2,
  SVE_ERR_OVERFLOW = -3,
  SVE_ERR_CORRUPT = -4,
  SVE_ERR_LIMIT = -5,
  SVE_ERR_NOTFOUND = -6,
  SVE_ERR_SQL = -7
} sve_status_t;

typedef enum sve_elem_type {
  SVE_ELEM_FLOAT32 = 223,
  SVE_ELEM_INT8 = 225,
  SVE_ELEM_BIT = 224
} sve_elem_type_t;

typedef enum sve_metric {
  SVE_METRIC_L2 = 1,
  SVE_METRIC_COSINE = 2,
  SVE_METRIC_L1 = 3,
  SVE_METRIC_IP = 4,
  SVE_METRIC_HAMMING = 5
} sve_metric_t;

typedef enum sve_index_kind {
  SVE_INDEX_FLAT = 0,
  SVE_INDEX_HNSW = 1,
  SVE_INDEX_HNSW_SQ8 = 2
} sve_index_kind_t;

#endif /* SVE_CONFIG_H */
