#ifndef SVE_HYBRID_RANK_H
#define SVE_HYBRID_RANK_H

#include "sve_hnsw.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sve_hybrid_hit {
  int64_t rowid;
  double fused_score;
  double vec_distance;
  double bm25_raw;
  uint32_t vec_rank;
  uint32_t bm25_rank;
} sve_hybrid_hit_t;

/*
 * Reciprocal Rank Fusion:
 *   score(d) = 1/(rrf_k + rank_vec) + 1/(rrf_k + rank_bm25)
 * Ranks are 1-based; candidates absent from one list contribute 0 for that list.
 * Output is ordered descending by fused_score with deterministic rowid tie-break.
 */
int sve_fuse_rrf(const sve_search_hit_t *vec_hits, size_t n_vec,
                 const sve_search_hit_t *bm25_hits, size_t n_bm25,
                 uint32_t rrf_k, size_t top_k, sve_hybrid_hit_t *out_hits,
                 size_t *out_count);

/*
 * Convex min-max calibrated fusion:
 *   sim_vec  = normalized ascending (lower distance => higher similarity in [0,1])
 *   sim_bm25 = normalized descending (-bm25 => higher relevance in [0,1])
 *   fused    = alpha * sim_vec + (1.0 - alpha) * sim_bm25
 */
int sve_fuse_convex(const sve_search_hit_t *vec_hits, size_t n_vec,
                    const sve_search_hit_t *bm25_hits, size_t n_bm25,
                    double alpha, size_t top_k, sve_hybrid_hit_t *out_hits,
                    size_t *out_count);

/*
 * Sanitizes free-form query strings into safe SQLite FTS5 MATCH expressions
 * (extracts alphanumeric / UTF-8 >= 0x80 tokens joined with " OR ").
 */
int sve_fts5_sanitize_query(const char *raw_query, size_t max_terms,
                            char *out_buf, size_t out_cap);

#ifdef __cplusplus
}
#endif

#endif /* SVE_HYBRID_RANK_H */
