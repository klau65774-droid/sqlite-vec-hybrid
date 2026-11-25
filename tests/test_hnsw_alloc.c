#include "test_common.h"
#include "sve_hybrid/sve_sqlite_ext.h"
#include <stdbool.h>

static size_t calls;
static size_t fail_at;
static size_t outstanding;

static bool fail_this_allocation(void) {
  calls++;
  return fail_at != 0U && calls == fail_at;
}

static void *tracked_malloc(size_t n) {
  void *ptr = fail_this_allocation() ? NULL : malloc(n);
  outstanding += ptr != NULL;
  return ptr;
}

static void *tracked_calloc(size_t n, size_t sz) {
  void *ptr = fail_this_allocation() ? NULL : calloc(n, sz);
  outstanding += ptr != NULL;
  return ptr;
}

static void *tracked_realloc(void *old, size_t n) {
  bool fresh = old == NULL;
  void *ptr = fail_this_allocation() ? NULL : realloc(old, n);
  outstanding += fresh && ptr != NULL;
  return ptr;
}

static void tracked_free(void *ptr) {
  if (ptr) {
    SVE_ASSERT(outstanding != 0U);
    outstanding--;
    free(ptr);
  }
}

/* Compile the production index into this test with an isolated allocator. */
#define malloc tracked_malloc
#define calloc tracked_calloc
#define realloc tracked_realloc
#define free tracked_free
#define sve_hnsw_serialize tracked_serialize
#include "../src/sve_hnsw.c"
#undef sve_hnsw_serialize
#undef malloc
#undef calloc
#undef realloc
#undef free

int sve_hnsw_serialize(const sve_hnsw_t *idx, uint8_t **out, size_t *len) {
  int rc = tracked_serialize(idx, out, len);
  if (rc == SVE_OK) {
    /* Snapshot ownership transfers to the caller, which uses ordinary free. */
    SVE_ASSERT(outstanding != 0U);
    outstanding--;
  }
  return rc;
}

static sve_hnsw_t *populated(uint32_t count) {
  sve_hnsw_config_t cfg =
      sve_hnsw_default_config(2U, SVE_ELEM_FLOAT32, SVE_METRIC_L2, SVE_INDEX_HNSW);
  cfg.m = 4U;
  cfg.rng_seed = 42ULL;
  sve_hnsw_t *idx = sve_hnsw_create(&cfg);
  SVE_ASSERT(idx);
  for (uint32_t i = 0U; i < count; i++) {
    float v[] = {(float)i, (float)(i % 11U)};
    SVE_ASSERT_EQ_INT(sve_hnsw_insert(idx, (int64_t)i, v), SVE_OK);
  }
  return idx;
}

static void assert_snapshot(const sve_hnsw_t *idx, const uint8_t *expected,
                            size_t expected_len) {
  uint8_t *actual = NULL;
  size_t length = 0U;
  SVE_ASSERT_EQ_INT(sve_hnsw_serialize(idx, &actual, &length), SVE_OK);
  SVE_ASSERT_EQ_INT(length, expected_len);
  SVE_ASSERT(memcmp(actual, expected, length) == 0);
  free(actual);
}

static void test_insert_oom(void) {
  const uint32_t counts[] = {1U, 64U, 95U, 128U};
  size_t failures = 0U;
  for (size_t i = 0U; i < sizeof(counts) / sizeof(counts[0]); i++) {
    uint32_t count = counts[i];
    const float v[] = {24.0f, 53.0f};
    sve_hnsw_t *reference = populated(count);
    uint8_t *before = NULL, *after = NULL;
    size_t before_len = 0U, after_len = 0U;
    SVE_ASSERT_EQ_INT(sve_hnsw_serialize(reference, &before, &before_len), SVE_OK);
    SVE_ASSERT_EQ_INT(sve_hnsw_insert(reference, (int64_t)count, v), SVE_OK);
    SVE_ASSERT_EQ_INT(sve_hnsw_serialize(reference, &after, &after_len), SVE_OK);
    sve_hnsw_destroy(reference);

    for (size_t at = 1U;; at++) {
      SVE_ASSERT(at <= 32U);
      sve_hnsw_t *idx = populated(count);
      fail_at = calls + at;
      int rc = sve_hnsw_insert(idx, (int64_t)count, v);
      fail_at = 0U;
      if (rc == SVE_OK) {
        assert_snapshot(idx, after, after_len);
        sve_hnsw_destroy(idx);
        break;
      }
      SVE_ASSERT_EQ_INT(rc, SVE_ERR_NOMEM);
      failures++;
      SVE_ASSERT(!sve_hnsw_contains(idx, (int64_t)count));
      SVE_ASSERT_EQ_INT(sve_hnsw_live_count(idx), count);
      assert_snapshot(idx, before, before_len);
      SVE_ASSERT_EQ_INT(sve_hnsw_insert(idx, (int64_t)count, v), SVE_OK);
      assert_snapshot(idx, after, after_len);
      sve_hnsw_destroy(idx);
    }
    free(before);
    free(after);
    SVE_ASSERT_EQ_INT(outstanding, 0U);
  }
  SVE_ASSERT(failures >= 14U);
  printf("Verified %zu insertion allocation failure points and deterministic retries\n",
         failures);
}

static void test_legacy(void) {
  /* Version 1 from the original binary after inserting two rows and deleting
     the first. Its surviving layer-0 edge is a historical tombstone reference. */
  const uint8_t snapshot[] = {
      0x53, 0x56, 0x45, 0x48, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0xdf, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x10, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
      0x81, 0x4b, 0xe3, 0x46, 0xb8, 0x49, 0xc2, 0x3c, 0x02, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f, 0x02, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x15, 0x52, 0x57, 0x01,
  };
  sve_hnsw_t *idx = NULL;
  SVE_ASSERT_EQ_INT(sve_hnsw_deserialize(snapshot, sizeof(snapshot), &idx), SVE_OK);
  SVE_ASSERT_EQ_INT(sve_hnsw_live_count(idx), 1U);
  SVE_ASSERT(!sve_hnsw_contains(idx, 1) && sve_hnsw_contains(idx, 2));
  uint8_t *new_snapshot = NULL;
  size_t len = 0U;
  SVE_ASSERT_EQ_INT(sve_hnsw_serialize(idx, &new_snapshot, &len), SVE_OK);
  SVE_ASSERT(len < sizeof(snapshot));
  sve_hnsw_destroy(idx);
  SVE_ASSERT_EQ_INT(sve_hnsw_deserialize(new_snapshot, len, &idx), SVE_OK);
  const float query[] = {2.0f};
  sve_search_hit_t hit[1];
  size_t n = 0U;
  SVE_ASSERT_EQ_INT(sve_hnsw_search(idx, query, 1U, 0U, hit, &n), SVE_OK);
  SVE_ASSERT_EQ_INT(n, 1U);
  SVE_ASSERT_EQ_INT(hit[0].rowid, 2);
  SVE_ASSERT_NEAR(hit[0].score_or_dist, 0.0, 0.0);
  free(new_snapshot);
  sve_hnsw_destroy(idx);
  SVE_ASSERT_EQ_INT(outstanding, 0U);
}

static void test_entry_and_allowlist(void) {
  sve_hnsw_t *idx = populated(400U);
  int64_t allowed[400];
  for (size_t i = 0U; i < sizeof(allowed) / sizeof(allowed[0]); i++) {
    allowed[i] = (int64_t)i;
  }
  for (int64_t i = 0; i < 400; i += 2) {
    SVE_ASSERT_EQ_INT(sve_hnsw_delete(idx, i), SVE_OK);
  }
  const float q[] = {110.0f, 4.0f};
  sve_search_hit_t *hits = (sve_search_hit_t *)malloc(SVE_MAX_K * sizeof(*hits));
  SVE_ASSERT(hits);
  size_t n = 0U;
  SVE_ASSERT_EQ_INT(sve_hnsw_search_allowlist(idx, q, 0U, UINT32_MAX, allowed,
                                             400U, NULL, &n), SVE_OK);
  SVE_ASSERT_EQ_INT(n, 0U);
  SVE_ASSERT_EQ_INT(sve_hnsw_search_allowlist(idx, q, 1U, UINT32_MAX, allowed,
                                             400U, hits, &n), SVE_OK);
  SVE_ASSERT_EQ_INT(n, 1U);
  SVE_ASSERT(hits[0].rowid % 2 == 1);
  SVE_ASSERT_EQ_INT(sve_hnsw_search_allowlist(idx, q, SVE_MAX_K, UINT32_MAX,
                                             allowed, 400U, hits, &n), SVE_OK);
  SVE_ASSERT_EQ_INT(n, 200U);
  for (size_t i = 0U; i < n; i++) {
    SVE_ASSERT(hits[i].rowid % 2 == 1);
  }
  uint32_t level = idx->max_level;
  while (idx->entry_point != UINT32_MAX) {
    int64_t id = idx->nodes[idx->entry_point].rowid;
    SVE_ASSERT_EQ_INT(sve_hnsw_delete(idx, id), SVE_OK);
    SVE_ASSERT(idx->max_level <= level);
    level = idx->max_level;
    uint8_t *serialized = NULL;
    size_t len = 0U;
    sve_hnsw_t *loaded = NULL;
    SVE_ASSERT_EQ_INT(sve_hnsw_serialize(idx, &serialized, &len), SVE_OK);
    SVE_ASSERT_EQ_INT(sve_hnsw_deserialize(serialized, len, &loaded), SVE_OK);
    SVE_ASSERT_EQ_INT(loaded->entry_point, idx->entry_point);
    SVE_ASSERT_EQ_INT(loaded->max_level, idx->max_level);
    free(serialized);
    sve_hnsw_destroy(loaded);
  }
  SVE_ASSERT_EQ_INT(sve_hnsw_live_count(idx), 0U);
  SVE_ASSERT_EQ_INT(sve_hnsw_insert(idx, INT64_MIN, q), SVE_OK);
  SVE_ASSERT_EQ_INT(sve_hnsw_search(idx, q, 1U, 0U, hits, &n), SVE_OK);
  SVE_ASSERT_EQ_INT(n, 1U);
  SVE_ASSERT(hits[0].rowid == INT64_MIN);
  free(hits);
  sve_hnsw_destroy(idx);
  SVE_ASSERT_EQ_INT(outstanding, 0U);
}

static int first_int(sqlite3 *db, const char *sql) {
  sqlite3_stmt *stmt = NULL;
  SVE_ASSERT_EQ_INT(sqlite3_prepare_v2(db, sql, -1, &stmt, NULL), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_step(stmt), SQLITE_ROW);
  int value = sqlite3_column_int(stmt, 0);
  SVE_ASSERT_EQ_INT(sqlite3_step(stmt), SQLITE_DONE);
  sqlite3_finalize(stmt);
  return value;
}

static void test_vtab_hnsw_oom(void) {
  sqlite3 *db = NULL;
  SVE_ASSERT_EQ_INT(sqlite3_open(":memory:", &db), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sve_hybrid_register(db, NULL), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_exec(db,
      "CREATE VIRTUAL TABLE v USING sve0(e float[1], body text fulltext);"
      "INSERT INTO v(rowid,e,body) VALUES(1,'[1]','original');"
      "BEGIN; INSERT INTO v(rowid,e,body) VALUES(2,'[2]','earlier');",
      NULL, NULL, NULL), SQLITE_OK);
  fail_at = calls + 1U;
  SVE_ASSERT_EQ_INT(sqlite3_exec(db,
      "INSERT OR REPLACE INTO v(rowid,e,body) VALUES(1,'[9]','edited');",
      NULL, NULL, NULL), SQLITE_NOMEM);
  fail_at = 0U;
  SVE_ASSERT_EQ_INT(first_int(db, "SELECT count(*) FROM v_rows WHERE id=1 AND c0='original';"),
                    1);
  SVE_ASSERT_EQ_INT(first_int(db, "SELECT rowid FROM v_fts WHERE v_fts MATCH 'original';"),
                    1);
  SVE_ASSERT_EQ_INT(first_int(db, "SELECT count(*) FROM v_fts WHERE v_fts MATCH 'edited';"),
                    0);
  SVE_ASSERT_EQ_INT(first_int(db, "SELECT rowid FROM v WHERE e MATCH '[1]' AND k=1;"), 1);
  if (!sqlite3_get_autocommit(db)) {
    SVE_ASSERT_EQ_INT(first_int(db, "SELECT count(*) FROM v_rows WHERE id=2;"), 1);
    SVE_ASSERT_EQ_INT(sqlite3_exec(db, "COMMIT;", NULL, NULL, NULL), SQLITE_OK);
  }
  SVE_ASSERT_EQ_INT(sqlite3_close(db), SQLITE_OK);
  SVE_ASSERT_EQ_INT(outstanding, 0U);
}

int main(int argc, char **argv) {
  SVE_ASSERT(argc == 2);
  if (strcmp(argv[1], "insert") == 0) {
    test_insert_oom();
  } else if (strcmp(argv[1], "legacy") == 0) {
    test_legacy();
  } else if (strcmp(argv[1], "entry") == 0) {
    test_entry_and_allowlist();
  } else if (strcmp(argv[1], "vtab") == 0) {
    test_vtab_hnsw_oom();
  } else {
    SVE_ASSERT(false);
  }
  return 0;
}
