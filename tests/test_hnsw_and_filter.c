#include "test_common.h"
#include "sqlite3.h"
#include "sve_hybrid/sve_hnsw.h"
#include "sve_hybrid/sve_numeric.h"
#include "sve_hybrid/sve_sqlite_ext.h"

#include <stdint.h>
#include <stdio.h>

static uint64_t lcg_next(uint64_t *s) {
  *s = (*s * 6364136223846793005ULL) + 1442695040888963407ULL;
  return *s;
}

static float lcg_f32(uint64_t *s) {
  uint64_t r = lcg_next(s);
  return (float)(((double)((r >> 16U) & 0xFFFFU) / 65536.0) * 2.0 - 1.0);
}

static void make_clustered_data(size_t n, size_t dims, float *vecs,
                                int64_t *cats, size_t n_cats, uint64_t seed) {
  uint64_t rng = seed;
  size_t nc = 16U;
  float *centers = (float *)malloc(nc * dims * sizeof(float));
  SVE_ASSERT(centers != NULL);
  for (size_t i = 0U; i < nc * dims; i++) {
    centers[i] = lcg_f32(&rng);
  }
  for (size_t i = 0U; i < n; i++) {
    const float *c = centers + (i % nc) * dims;
    double norm = 0.0;
    for (size_t d = 0U; d < dims; d++) {
      float v = c[d] + 0.2f * lcg_f32(&rng);
      vecs[i * dims + d] = v;
      norm += (double)v * (double)v;
    }
    double inv = norm > 0.0 ? 1.0 / sqrt(norm) : 1.0;
    for (size_t d = 0U; d < dims; d++) {
      vecs[i * dims + d] = (float)((double)vecs[i * dims + d] * inv);
    }
    cats[i] = (int64_t)(i % n_cats);
  }
  free(centers);
}

static double recall_at_k(const sve_search_hit_t *gt, size_t ng,
                          const sve_search_hit_t *pr, size_t np) {
  size_t hits = 0U;
  for (size_t i = 0U; i < np; i++) {
    for (size_t j = 0U; j < ng; j++) {
      if (pr[i].rowid == gt[j].rowid) {
        hits++;
        break;
      }
    }
  }
  return ng > 0U ? (double)hits / (double)ng : 1.0;
}

static void test_recall_1k(void) {
  const size_t n = 1000U, dims = 64U, k = 10U, nq = 25U;
  float *vecs = (float *)malloc(n * dims * sizeof(float));
  int64_t *cats = (int64_t *)malloc(n * sizeof(int64_t));
  SVE_ASSERT(vecs && cats);
  make_clustered_data(n, dims, vecs, cats, 10U, 1337ULL);

  sve_hnsw_config_t cfg_flat = sve_hnsw_default_config(
      dims, SVE_ELEM_FLOAT32, SVE_METRIC_COSINE, SVE_INDEX_FLAT);
  sve_hnsw_config_t cfg_hnsw = sve_hnsw_default_config(
      dims, SVE_ELEM_FLOAT32, SVE_METRIC_COSINE, SVE_INDEX_HNSW);
  sve_hnsw_t *exact = sve_hnsw_create(&cfg_flat);
  sve_hnsw_t *hnsw = sve_hnsw_create(&cfg_hnsw);
  SVE_ASSERT(exact && hnsw);

  for (size_t i = 0U; i < n; i++) {
    int64_t rid = (int64_t)(i + 1U);
    SVE_ASSERT_EQ_INT(sve_hnsw_insert(exact, rid, vecs + i * dims), SVE_OK);
    SVE_ASSERT_EQ_INT(sve_hnsw_insert(hnsw, rid, vecs + i * dims), SVE_OK);
  }

  double sum_rec = 0.0;
  for (size_t q = 0U; q < nq; q++) {
    sve_search_hit_t gt[10], pr[10];
    size_t ng = 0U, np = 0U;
    SVE_ASSERT_EQ_INT(
        sve_hnsw_exact_search(exact, vecs + q * 37U * dims, k, NULL, 0U, gt, &ng),
        SVE_OK);
    SVE_ASSERT_EQ_INT(
        sve_hnsw_search(hnsw, vecs + q * 37U * dims, k, 64U, pr, &np), SVE_OK);
    SVE_ASSERT_EQ_INT(np, k);
    sum_rec += recall_at_k(gt, ng, pr, np);
  }
  double mean_rec = sum_rec / (double)nq;
  printf("[HNSW 1K D64] Mean Recall@10 = %.4f\n", mean_rec);
  SVE_ASSERT(mean_rec >= 0.95);

  sve_hnsw_destroy(exact);
  sve_hnsw_destroy(hnsw);
  free(vecs);
  free(cats);
}

static void test_sq8_quantized(void) {
  const size_t n = 600U, dims = 64U, k = 10U, nq = 20U;
  float *vecs = (float *)malloc(n * dims * sizeof(float));
  int64_t *cats = (int64_t *)malloc(n * sizeof(int64_t));
  SVE_ASSERT(vecs && cats);
  make_clustered_data(n, dims, vecs, cats, 5U, 99ULL);

  sve_hnsw_config_t cfg_flat = sve_hnsw_default_config(
      dims, SVE_ELEM_FLOAT32, SVE_METRIC_COSINE, SVE_INDEX_FLAT);
  sve_hnsw_config_t cfg_sq8 = sve_hnsw_default_config(
      dims, SVE_ELEM_FLOAT32, SVE_METRIC_COSINE, SVE_INDEX_HNSW_SQ8);
  sve_hnsw_t *exact = sve_hnsw_create(&cfg_flat);
  sve_hnsw_t *sq8 = sve_hnsw_create(&cfg_sq8);
  SVE_ASSERT(exact && sq8);

  for (size_t i = 0U; i < n; i++) {
    int64_t rid = (int64_t)(i + 1U);
    SVE_ASSERT_EQ_INT(sve_hnsw_insert(exact, rid, vecs + i * dims), SVE_OK);
    SVE_ASSERT_EQ_INT(sve_hnsw_insert(sq8, rid, vecs + i * dims), SVE_OK);
  }

  double sum_rec = 0.0;
  for (size_t q = 0U; q < nq; q++) {
    sve_search_hit_t gt[10], pr[10];
    size_t ng = 0U, np = 0U;
    SVE_ASSERT_EQ_INT(
        sve_hnsw_exact_search(exact, vecs + q * 23U * dims, k, NULL, 0U, gt, &ng),
        SVE_OK);
    SVE_ASSERT_EQ_INT(
        sve_hnsw_search(sq8, vecs + q * 23U * dims, k, 64U, pr, &np), SVE_OK);
    sum_rec += recall_at_k(gt, ng, pr, np);
  }
  double mean_rec = sum_rec / (double)nq;
  printf("[HNSW_SQ8 600 D64] Mean Recall@10 = %.4f\n", mean_rec);
  SVE_ASSERT(mean_rec >= 0.90);

  sve_hnsw_destroy(exact);
  sve_hnsw_destroy(sq8);
  free(vecs);
  free(cats);
}

static void test_selective_prefilter(void) {
  /* 1% predicate selectivity across 1000 rows via SQLite WHERE filter */
  const size_t n = 1000U, dims = 32U;
  float *vecs = (float *)malloc(n * dims * sizeof(float));
  int64_t *cats = (int64_t *)malloc(n * sizeof(int64_t));
  SVE_ASSERT(vecs && cats);
  make_clustered_data(n, dims, vecs, cats, 100U, 2026ULL);

  sqlite3 *db = NULL;
  SVE_ASSERT_EQ_INT(sqlite3_open(":memory:", &db), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sve_hybrid_register(db, NULL), SQLITE_OK);
  SVE_ASSERT_EQ_INT(
      sqlite3_exec(db,
                   "CREATE VIRTUAL TABLE tf USING sve0("
                   "  emb float[32] index=hnsw metric=cosine,"
                   "  bucket integer"
                   ");",
                   NULL, NULL, NULL),
      SQLITE_OK);

  SVE_ASSERT_EQ_INT(sqlite3_exec(db, "BEGIN;", NULL, NULL, NULL), SQLITE_OK);
  sqlite3_stmt *ins = NULL;
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db, "INSERT INTO tf(rowid, emb, bucket) VALUES (?1, ?2, ?3);", -1,
          &ins, NULL),
      SQLITE_OK);
  for (size_t i = 0U; i < n; i++) {
    sqlite3_reset(ins);
    sqlite3_bind_int64(ins, 1, (sqlite3_int64)(i + 1U));
    sqlite3_bind_blob(ins, 2, vecs + i * dims, (int)(dims * sizeof(float)),
                      SQLITE_STATIC);
    sqlite3_bind_int64(ins, 3, cats[i]);
    SVE_ASSERT_EQ_INT(sqlite3_step(ins), SQLITE_DONE);
  }
  sqlite3_finalize(ins);
  SVE_ASSERT_EQ_INT(sqlite3_exec(db, "COMMIT;", NULL, NULL, NULL), SQLITE_OK);

  /* Query bucket=7 (exactly 10 rows = 1% selectivity) asking for k=10 */
  sqlite3_stmt *q = NULL;
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db,
          "SELECT rowid, bucket FROM tf "
          "WHERE emb MATCH ?1 AND k = 10 AND bucket = 7;",
          -1, &q, NULL),
      SQLITE_OK);
  sqlite3_bind_blob(q, 1, vecs, (int)(dims * sizeof(float)), SQLITE_STATIC);
  size_t count = 0U;
  while (sqlite3_step(q) == SQLITE_ROW) {
    SVE_ASSERT_EQ_INT(sqlite3_column_int64(q, 1), 7);
    count++;
  }
  /* Pre-filter fallback guarantees 10/10 hits instead of post-filter starvation */
  SVE_ASSERT_EQ_INT(count, 10U);
  sqlite3_finalize(q);
  sqlite3_close(db);
  free(vecs);
  free(cats);
}

static void test_crc_corrupt(void) {
  sve_hnsw_config_t cfg = sve_hnsw_default_config(
      8U, SVE_ELEM_FLOAT32, SVE_METRIC_L2, SVE_INDEX_HNSW);
  sve_hnsw_t *idx = sve_hnsw_create(&cfg);
  SVE_ASSERT(idx != NULL);
  float v[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  SVE_ASSERT_EQ_INT(sve_hnsw_insert(idx, 101, v), SVE_OK);
  v[0] = -3.0f;
  SVE_ASSERT_EQ_INT(sve_hnsw_insert(idx, 102, v), SVE_OK);

  uint8_t *blob = NULL;
  size_t blen = 0U;
  SVE_ASSERT_EQ_INT(sve_hnsw_serialize(idx, &blob, &blen), SVE_OK);
  SVE_ASSERT(blen > 64U);

  sve_hnsw_t *restored = NULL;
  SVE_ASSERT_EQ_INT(sve_hnsw_deserialize(blob, blen, &restored), SVE_OK);
  SVE_ASSERT_EQ_INT(sve_hnsw_live_count(restored), 2U);
  sve_hnsw_destroy(restored);

  /* Flip 1 bit in payload -> CRC32C must reject cleanly */
  blob[blen / 2U] ^= 0x01U;
  SVE_ASSERT_EQ_INT(sve_hnsw_deserialize(blob, blen, &restored),
                    SVE_ERR_CORRUPT);
  SVE_ASSERT(restored == NULL);

  free(blob);
  sve_hnsw_destroy(idx);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    test_recall_1k();
    test_sq8_quantized();
    test_selective_prefilter();
    test_crc_corrupt();
    return 0;
  }
  if (strcmp(argv[1], "recall_1k") == 0) {
    test_recall_1k();
  } else if (strcmp(argv[1], "sq8") == 0) {
    test_sq8_quantized();
  } else if (strcmp(argv[1], "filter") == 0) {
    test_selective_prefilter();
  } else if (strcmp(argv[1], "crc_corrupt") == 0) {
    test_crc_corrupt();
  } else {
    return 1;
  }
  return 0;
}
