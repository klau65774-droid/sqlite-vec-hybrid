#ifndef SVE_HNSW_H
#define SVE_HNSW_H

#include "sve_config.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sve_hnsw_config {
  size_t dims;
  sve_elem_type_t elem_type;
  sve_metric_t metric;
  sve_index_kind_t index_kind;
  uint32_t m;
  uint32_t ef_construction;
  uint32_t ef_search;
  uint64_t rng_seed;
} sve_hnsw_config_t;

typedef struct sve_search_hit {
  int64_t rowid;
  double score_or_dist;
} sve_search_hit_t;

typedef struct sve_hnsw sve_hnsw_t;

sve_hnsw_config_t sve_hnsw_default_config(size_t dims,
                                          sve_elem_type_t elem_type,
                                          sve_metric_t metric,
                                          sve_index_kind_t index_kind);
bool sve_hnsw_config_is_valid(const sve_hnsw_config_t *cfg);

sve_hnsw_t *sve_hnsw_create(const sve_hnsw_config_t *cfg);
void sve_hnsw_destroy(sve_hnsw_t *idx);

/* Float32 vectors passed to insert or search must be arrays of float. */
int sve_hnsw_insert(sve_hnsw_t *idx, int64_t rowid, const void *vector_data);
int sve_hnsw_delete(sve_hnsw_t *idx, int64_t rowid);
bool sve_hnsw_contains(const sve_hnsw_t *idx, int64_t rowid);
size_t sve_hnsw_live_count(const sve_hnsw_t *idx);

int sve_hnsw_search(const sve_hnsw_t *idx, const void *query_vec, size_t k,
                    uint32_t ef_override, sve_search_hit_t *out_hits,
                    size_t *out_count);

/*
 * Adaptive metadata-filtered KNN search:
 * - allowed_rowids_sorted must be ascending unique SQLite rowids.
 * - Switches cleanly to exact filtered evaluation when n_allowed <=
 *   SVE_EXACT_FILTER_THRESHOLD, and traverses all live graph bridges with
 *   boosted ef + guaranteed top-up when n_allowed > threshold.
 */
int sve_hnsw_search_allowlist(const sve_hnsw_t *idx, const void *query_vec,
                              size_t k, uint32_t ef_override,
                              const int64_t *allowed_rowids_sorted,
                              size_t n_allowed, sve_search_hit_t *out_hits,
                              size_t *out_count);

/* Linear exact baseline scan over all live entries (for testing/FLAT mode) */
int sve_hnsw_exact_search(const sve_hnsw_t *idx, const void *query_vec,
                          size_t k, const int64_t *allowed_rowids_sorted,
                          size_t n_allowed, sve_search_hit_t *out_hits,
                          size_t *out_count);

/* Binary serialization framed with SVEH header and trailing CRC32C checksum */
int sve_hnsw_serialize(const sve_hnsw_t *idx, uint8_t **out_buf,
                       size_t *out_len);
int sve_hnsw_deserialize(const uint8_t *buf, size_t len, sve_hnsw_t **out_idx);
/* Constrain the snapshot to the caller's index definition. The seed is ignored. */
int sve_hnsw_deserialize_for_config(const uint8_t *buf, size_t len,
                                    const sve_hnsw_config_t *expected,
                                    sve_hnsw_t **out_idx);

#ifdef __cplusplus
}
#endif

#endif /* SVE_HNSW_H */
