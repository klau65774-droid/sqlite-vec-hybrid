#include "sve_hybrid/sve_hybrid_rank.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

typedef struct rank_entry {
  int64_t rowid;
  double vec_dist;
  double bm25_raw;
  uint32_t vec_rank;
  uint32_t bm25_rank;
  bool has_vec;
  bool has_bm25;
  double score;
} rank_entry_t;

typedef struct rowid_map_slot {
  int64_t key;
  size_t entry_idx;
  bool occupied;
} rowid_map_slot_t;

static uint64_t hash_i64(int64_t v) {
  uint64_t x = (uint64_t)v;
  x ^= x >> 30U;
  x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 27U;
  x *= 0x94D049BB133111EBULL;
  x ^= x >> 31U;
  return x;
}

static size_t next_pow2_min16(size_t n) {
  size_t c = 16U;
  while (c < n) {
    if (c > SIZE_MAX / (2U * sizeof(rowid_map_slot_t))) {
      return 0U;
    }
    c *= 2U;
  }
  return c;
}

static int find_or_insert_entry(rowid_map_slot_t *slots, size_t mask,
                                rank_entry_t *entries, size_t *entry_count,
                                int64_t rowid, size_t *out_idx) {
  uint64_t h = hash_i64(rowid);
  size_t pos = (size_t)h & mask;
  for (;;) {
    if (!slots[pos].occupied) {
      size_t idx = (*entry_count)++;
      memset(&entries[idx], 0, sizeof(rank_entry_t));
      entries[idx].rowid = rowid;
      slots[pos].occupied = true;
      slots[pos].key = rowid;
      slots[pos].entry_idx = idx;
      *out_idx = idx;
      return SVE_OK;
    }
    if (slots[pos].key == rowid) {
      *out_idx = slots[pos].entry_idx;
      return SVE_OK;
    }
    pos = (pos + 1U) & mask;
  }
}

static int cmp_hybrid_desc(const void *pa, const void *pb) {
  const sve_hybrid_hit_t *a = (const sve_hybrid_hit_t *)pa;
  const sve_hybrid_hit_t *b = (const sve_hybrid_hit_t *)pb;
  if (a->fused_score > b->fused_score) {
    return -1;
  }
  if (a->fused_score < b->fused_score) {
    return 1;
  }
  if (a->rowid < b->rowid) {
    return -1;
  }
  if (a->rowid > b->rowid) {
    return 1;
  }
  return 0;
}

static int merge_candidates(const sve_search_hit_t *vec_hits, size_t n_vec,
                            const sve_search_hit_t *bm25_hits, size_t n_bm25,
                            rank_entry_t **out_entries, size_t *out_count) {
  *out_entries = NULL;
  *out_count = 0U;
  if (n_vec > UINT32_MAX || n_bm25 > UINT32_MAX || n_vec > SIZE_MAX - n_bm25) {
    return SVE_ERR_LIMIT;
  }
  size_t total = n_vec + n_bm25;
  if (total == 0U) {
    return SVE_OK;
  }
  if (total > (SIZE_MAX - 4U) / 2U || total > SIZE_MAX / sizeof(rank_entry_t) ||
      total > SIZE_MAX / sizeof(sve_hybrid_hit_t)) {
    return SVE_ERR_LIMIT;
  }
  size_t map_cap = next_pow2_min16(total * 2U + 4U);
  if (map_cap == 0U) {
    return SVE_ERR_LIMIT;
  }
  rank_entry_t *entries = (rank_entry_t *)calloc(total, sizeof(*entries));
  rowid_map_slot_t *slots =
      (rowid_map_slot_t *)calloc(map_cap, sizeof(rowid_map_slot_t));
  if (!entries || !slots) {
    free(entries);
    free(slots);
    return SVE_ERR_NOMEM;
  }

  size_t mask = map_cap - 1U;
  size_t count = 0U;

  for (size_t i = 0; i < n_vec; i++) {
    size_t idx = 0U;
    (void)find_or_insert_entry(slots, mask, entries, &count, vec_hits[i].rowid,
                               &idx);
    if (!entries[idx].has_vec) {
      entries[idx].has_vec = true;
      entries[idx].vec_dist = vec_hits[i].score_or_dist;
      entries[idx].vec_rank = (uint32_t)(i + 1U);
    }
  }
  for (size_t i = 0; i < n_bm25; i++) {
    size_t idx = 0U;
    (void)find_or_insert_entry(slots, mask, entries, &count, bm25_hits[i].rowid,
                               &idx);
    if (!entries[idx].has_bm25) {
      entries[idx].has_bm25 = true;
      entries[idx].bm25_raw = bm25_hits[i].score_or_dist;
      entries[idx].bm25_rank = (uint32_t)(i + 1U);
    }
  }

  free(slots);
  *out_entries = entries;
  *out_count = count;
  return SVE_OK;
}

static int finalize_hits(const rank_entry_t *entries, size_t count,
                         size_t top_k, sve_hybrid_hit_t *out_hits,
                         size_t *out_count) {
  sve_hybrid_hit_t *all =
      (sve_hybrid_hit_t *)calloc(count, sizeof(sve_hybrid_hit_t));
  if (!all) {
    return SVE_ERR_NOMEM;
  }
  for (size_t i = 0; i < count; i++) {
    all[i].rowid = entries[i].rowid;
    all[i].fused_score = entries[i].score;
    all[i].vec_distance = entries[i].vec_dist;
    all[i].bm25_raw = entries[i].bm25_raw;
    all[i].vec_rank = entries[i].vec_rank;
    all[i].bm25_rank = entries[i].bm25_rank;
  }
  qsort(all, count, sizeof(sve_hybrid_hit_t), cmp_hybrid_desc);
  size_t emit = count < top_k ? count : top_k;
  if (emit > 0U) {
    memcpy(out_hits, all, emit * sizeof(sve_hybrid_hit_t));
  }
  free(all);
  *out_count = emit;
  return SVE_OK;
}

int sve_fuse_rrf(const sve_search_hit_t *vec_hits, size_t n_vec,
                 const sve_search_hit_t *bm25_hits, size_t n_bm25,
                 uint32_t rrf_k, size_t top_k, sve_hybrid_hit_t *out_hits,
                 size_t *out_count) {
  if (!out_count || (top_k > 0U && !out_hits) || rrf_k == 0U ||
      (n_vec > 0U && !vec_hits) || (n_bm25 > 0U && !bm25_hits)) {
    return SVE_ERR_INVALID;
  }
  *out_count = 0U;
  if (top_k == 0U || (n_vec == 0U && n_bm25 == 0U)) {
    return SVE_OK;
  }

  rank_entry_t *entries = NULL;
  size_t count = 0U;
  int rc = merge_candidates(vec_hits, n_vec, bm25_hits, n_bm25, &entries, &count);
  if (rc != SVE_OK) {
    return rc;
  }
  if (count == 0U) {
    free(entries);
    return SVE_OK;
  }

  double kd = (double)rrf_k;
  for (size_t i = 0; i < count; i++) {
    double s = 0.0;
    if (entries[i].has_vec) {
      s += 1.0 / (kd + (double)entries[i].vec_rank);
    }
    if (entries[i].has_bm25) {
      s += 1.0 / (kd + (double)entries[i].bm25_rank);
    }
    entries[i].score = s;
  }

  rc = finalize_hits(entries, count, top_k, out_hits, out_count);
  free(entries);
  return rc;
}

static double scale_for_span(double min, double max) {
  return min < 0.0 && max > 0.0 && max >= DBL_MAX + min ? 0.5 : 1.0;
}

int sve_fuse_convex(const sve_search_hit_t *vec_hits, size_t n_vec,
                    const sve_search_hit_t *bm25_hits, size_t n_bm25,
                    double alpha, size_t top_k, sve_hybrid_hit_t *out_hits,
                    size_t *out_count) {
  if (!out_count || (top_k > 0U && !out_hits) || alpha < 0.0 || alpha > 1.0 ||
      !isfinite(alpha) || (n_vec > 0U && !vec_hits) ||
      (n_bm25 > 0U && !bm25_hits)) {
    return SVE_ERR_INVALID;
  }
  *out_count = 0U;
  if (top_k == 0U || (n_vec == 0U && n_bm25 == 0U)) {
    return SVE_OK;
  }

  rank_entry_t *entries = NULL;
  size_t count = 0U;
  int rc = merge_candidates(vec_hits, n_vec, bm25_hits, n_bm25, &entries, &count);
  if (rc != SVE_OK) {
    return rc;
  }
  if (count == 0U) {
    free(entries);
    return SVE_OK;
  }

  double v_min = INFINITY, v_max = -INFINITY;
  double b_min = INFINITY, b_max = -INFINITY;
  for (size_t i = 0; i < count; i++) {
    if (entries[i].has_vec && isfinite(entries[i].vec_dist)) {
      if (entries[i].vec_dist < v_min) {
        v_min = entries[i].vec_dist;
      }
      if (entries[i].vec_dist > v_max) {
        v_max = entries[i].vec_dist;
      }
    }
    if (entries[i].has_bm25 && isfinite(entries[i].bm25_raw)) {
      /* SQLite FTS5 bm25() returns more negative for higher relevance */
      double rel = -entries[i].bm25_raw;
      if (rel < b_min) {
        b_min = rel;
      }
      if (rel > b_max) {
        b_max = rel;
      }
    }
  }

  double v_scale = scale_for_span(v_min, v_max);
  double b_scale = scale_for_span(b_min, b_max);
  v_min *= v_scale;
  b_min *= b_scale;
  double v_span = (isfinite(v_min) && isfinite(v_max)) ? (v_max * v_scale - v_min) : 0.0;
  double b_span = (isfinite(b_min) && isfinite(b_max)) ? (b_max * b_scale - b_min) : 0.0;

  for (size_t i = 0; i < count; i++) {
    double s_vec = 0.0;
    double s_bm25 = 0.0;
    if (entries[i].has_vec && isfinite(entries[i].vec_dist)) {
      s_vec = (v_span > 1e-12)
                  ? (1.0 - (entries[i].vec_dist * v_scale - v_min) / v_span)
                  : 1.0;
    }
    if (entries[i].has_bm25 && isfinite(entries[i].bm25_raw)) {
      double rel = -entries[i].bm25_raw;
      s_bm25 = (b_span > 1e-12) ? ((rel * b_scale - b_min) / b_span) : 1.0;
    }
    entries[i].score = alpha * s_vec + (1.0 - alpha) * s_bm25;
  }

  rc = finalize_hits(entries, count, top_k, out_hits, out_count);
  free(entries);
  return rc;
}

static bool valid_utf8(const unsigned char *s, size_t len) {
  size_t i = 0U;
  while (i < len) {
    unsigned char lead = s[i++];
    if (lead < 0x80U) {
      continue;
    }
    if (lead < 0xC2U || lead > 0xF4U) {
      return false;
    }
    size_t rest = lead < 0xE0U ? 1U : (lead < 0xF0U ? 2U : 3U);
    if (rest > len - i) {
      return false;
    }
    unsigned char second = s[i];
    if ((lead == 0xE0U && second < 0xA0U) ||
        (lead == 0xEDU && second >= 0xA0U) ||
        (lead == 0xF0U && second < 0x90U) ||
        (lead == 0xF4U && second >= 0x90U)) {
      return false;
    }
    for (size_t j = 0U; j < rest; j++) {
      if ((s[i++] & 0xC0U) != 0x80U) {
        return false;
      }
    }
  }
  return true;
}

static bool query_token_byte(unsigned char c) {
  return c >= 0x80U || c == '_' || (c >= '0' && c <= '9') ||
         (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

int sve_fts5_sanitize_query(const char *raw_query, size_t max_terms,
                            char *out_buf, size_t out_cap) {
  if (!out_buf || out_cap == 0U || max_terms == 0U) {
    return SVE_ERR_INVALID;
  }
  out_buf[0] = '\0';
  if (!raw_query) {
    return SVE_OK;
  }

  size_t raw_len = 0U;
  while (raw_len <= SVE_MAX_TEXT_BYTES && raw_query[raw_len] != '\0') {
    raw_len++;
  }
  if (raw_len > SVE_MAX_TEXT_BYTES) {
    return SVE_ERR_LIMIT;
  }
  if (!valid_utf8((const unsigned char *)raw_query, raw_len)) {
    return SVE_ERR_INVALID;
  }

  size_t pos = 0U;
  size_t out_pos = 0U;
  size_t terms = 0U;

  while (pos < raw_len && terms < max_terms) {
    if (!query_token_byte((unsigned char)raw_query[pos])) {
      pos++;
      continue;
    }
    size_t start = pos;
    while (pos < raw_len && query_token_byte((unsigned char)raw_query[pos])) {
      pos++;
    }
    size_t tok_len = pos - start;
    /* Prefix with OR between quoted tokens so FTS5 never throws syntax error */
    size_t sep_len = (terms > 0U) ? 4U : 0U;
    size_t needed = sep_len + tok_len + 2U; /* "..." */
    if (needed >= out_cap - out_pos) {
      break;
    }
    if (terms > 0U) {
      memcpy(out_buf + out_pos, " OR ", 4U);
      out_pos += 4U;
    }
    out_buf[out_pos++] = '"';
    memcpy(out_buf + out_pos, raw_query + start, tok_len);
    out_pos += tok_len;
    out_buf[out_pos++] = '"';
    out_buf[out_pos] = '\0';
    terms++;
  }
  return SVE_OK;
}
