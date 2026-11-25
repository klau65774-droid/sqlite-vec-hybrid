#include "test_common.h"
#include "sqlite3.h"
#include "sqlite-vec.h"

#include <stdint.h>

static void exec_ok(sqlite3 *db, const char *sql) {
  char *err = NULL;
  int rc = sqlite3_exec(db, sql, NULL, NULL, &err);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "SQL failed (%d): %s\nErr: %s\n", rc, sql,
            err ? err : "(null)");
    sqlite3_free(err);
    abort();
  }
}

/* Issue #333: failed explicit-txn INSERT leaves persistent searchable ghost row */
static void repro_issue_333_ghost_row(void) {
  sqlite3 *db = NULL;
  SVE_ASSERT_EQ_INT(sqlite3_open(":memory:", &db), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_vec_init(db, NULL, NULL), SQLITE_OK);

  exec_ok(db,
          "CREATE VIRTUAL TABLE t_ghost USING vec0("
          "  emb float[4],"
          "  +code integer"
          ");");
  exec_ok(db, "BEGIN;");
  /* Fails at auxiliary column type validation after _rowids & _chunks write */
  char *err = NULL;
  int rc = sqlite3_exec(
      db,
      "INSERT INTO t_ghost(rowid, emb, code) "
      "VALUES (42, '[1.0, 0.0, 0.0, 0.0]', 'invalid_int_text');",
      NULL, NULL, &err);
  SVE_ASSERT(rc != SQLITE_OK);
  sqlite3_free(err);
  exec_ok(db, "COMMIT;");

  sqlite3_stmt *stmt = NULL;
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db,
          "SELECT rowid, distance FROM t_ghost "
          "WHERE emb MATCH '[1.0, 0.0, 0.0, 0.0]' AND k = 5;",
          -1, &stmt, NULL),
      SQLITE_OK);
  int step_rc = sqlite3_step(stmt);
  /* Upstream bug manifests: rowid 42 was rejected yet appears in KNN results */
  SVE_ASSERT_EQ_INT(step_rc, SQLITE_ROW);
  sqlite3_int64 ghost_rowid = sqlite3_column_int64(stmt, 0);
  SVE_ASSERT_EQ_INT(ghost_rowid, 42);
  printf("[UPSTREAM REPRO #333] Confirmed ghost row leaked: rowid=%lld\n",
         (long long)ghost_rowid);
  sqlite3_finalize(stmt);
  sqlite3_close(db);
}

/* Issue #361: vec0 int8 columns reject valid untyped bound BLOBs */
static void repro_issue_361_int8_blob(void) {
  sqlite3 *db = NULL;
  SVE_ASSERT_EQ_INT(sqlite3_open(":memory:", &db), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_vec_init(db, NULL, NULL), SQLITE_OK);

  exec_ok(db, "CREATE VIRTUAL TABLE t_i8 USING vec0(v int8[4]);");
  sqlite3_stmt *q = NULL;
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(db, "SELECT vec_int8('[10, -20, 30, -40]');", -1, &q,
                         NULL),
      SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_step(q), SQLITE_ROW);
  const void *raw = sqlite3_column_blob(q, 0);
  int raw_bytes = sqlite3_column_bytes(q, 0);
  SVE_ASSERT_EQ_INT(raw_bytes, 4);

  sqlite3_stmt *ins = NULL;
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(db, "INSERT INTO t_i8(rowid, v) VALUES (1, ?1);", -1,
                         &ins, NULL),
      SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_bind_blob(ins, 1, raw, raw_bytes, SQLITE_TRANSIENT),
                    SQLITE_OK);
  int rc = sqlite3_step(ins);
  /* Upstream bug manifests: bound 4-byte int8 BLOB misclassified as float32 */
  SVE_ASSERT(rc != SQLITE_OK);
  const char *msg = sqlite3_errmsg(db);
  SVE_ASSERT(strstr(msg, "float32") != NULL);
  printf("[UPSTREAM REPRO #361] Confirmed bound int8 BLOB rejected: %s\n", msg);

  sqlite3_finalize(ins);
  sqlite3_finalize(q);
  sqlite3_close(db);
}

/* Issue #334: vec_distance_l2 returns Inf for 1e20 and 0 for 1e-25 */
static void repro_issue_334_l2_scale(void) {
  sqlite3 *db = NULL;
  SVE_ASSERT_EQ_INT(sqlite3_open(":memory:", &db), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_vec_init(db, NULL, NULL), SQLITE_OK);

  sqlite3_stmt *s1 = NULL;
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db, "SELECT vec_distance_l2('[1e20, 0, 0, 0]', '[0, 0, 0, 0]');", -1,
          &s1, NULL),
      SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_step(s1), SQLITE_ROW);
  double d_large = sqlite3_column_double(s1, 0);
  SVE_ASSERT(isinf(d_large));

  sqlite3_stmt *s2 = NULL;
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db, "SELECT vec_distance_l2('[1e-25, 0, 0, 0]', '[0, 0, 0, 0]');",
          -1, &s2, NULL),
      SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_step(s2), SQLITE_ROW);
  double d_tiny = sqlite3_column_double(s2, 0);
  SVE_ASSERT_EQ_INT(d_tiny == 0.0, 1);

  printf("[UPSTREAM REPRO #334] Confirmed L2 overflow=%.6g underflow=%.6g\n",
         d_large, d_tiny);
  sqlite3_finalize(s1);
  sqlite3_finalize(s2);
  sqlite3_close(db);
}

/* Issue #362 + #363: int8 L1 INT32 wrap & vec_add silent int8 wrap / Inf emit */
static void repro_issue_362_363_arith(void) {
  sqlite3 *db = NULL;
  SVE_ASSERT_EQ_INT(sqlite3_open(":memory:", &db), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_vec_init(db, NULL, NULL), SQLITE_OK);

  /* #363 int8 silent modulo wrap: 100 + 50 -> -106 */
  sqlite3_stmt *s_add = NULL;
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db,
          "SELECT vec_to_json(vec_add(vec_int8('[100, 100]'), "
          "vec_int8('[50, 50]')));",
          -1, &s_add, NULL),
      SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_step(s_add), SQLITE_ROW);
  const char *wrapped = (const char *)sqlite3_column_text(s_add, 0);
  SVE_ASSERT(wrapped != NULL && strstr(wrapped, "-106") != NULL);
  printf("[UPSTREAM REPRO #363] Confirmed vec_add int8 silent wrap: %s\n",
         wrapped);
  sqlite3_finalize(s_add);

  /* #362 L1 int8 overflow past INT32_MAX: build two 8,450,000-dim vectors */
  const size_t dims = 8450000U;
  /* "127," takes 4 bytes/elem; "-128," takes 5 bytes/elem */
  size_t ja_cap = dims * 4U + 16U;
  size_t jb_cap = dims * 5U + 16U;
  char *ja = (char *)malloc(ja_cap);
  char *jb = (char *)malloc(jb_cap);
  SVE_ASSERT(ja != NULL && jb != NULL);
  size_t pa = 0, pb = 0;
  ja[pa++] = '[';
  jb[pb++] = '[';
  for (size_t i = 0; i < dims; i++) {
    if (i > 0) {
      ja[pa++] = ',';
      jb[pb++] = ',';
    }
    memcpy(ja + pa, "127", 3);
    pa += 3;
    memcpy(jb + pb, "-128", 4);
    pb += 4;
  }
  ja[pa++] = ']';
  ja[pa] = '\0';
  jb[pb++] = ']';
  jb[pb] = '\0';

  sqlite3_stmt *s_l1 = NULL;
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db, "SELECT vec_distance_l1(vec_int8(?1), vec_int8(?2));", -1,
          &s_l1, NULL),
      SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_bind_text(s_l1, 1, ja, (int)pa, SQLITE_STATIC),
                    SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_bind_text(s_l1, 2, jb, (int)pb, SQLITE_STATIC),
                    SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_step(s_l1), SQLITE_ROW);
  sqlite3_int64 l1_res = sqlite3_column_int64(s_l1, 0);
  /* Upstream bug manifests: wrapped to negative value */
  SVE_ASSERT(l1_res < 0);
  printf("[UPSTREAM REPRO #362] Confirmed int8 L1 INT32 overflow: %lld\n",
         (long long)l1_res);

  sqlite3_finalize(s_l1);
  free(ja);
  free(jb);
  sqlite3_close(db);
}

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);
  fprintf(stderr, "-> running #333...\n");
  repro_issue_333_ghost_row();
  fprintf(stderr, "-> running #361...\n");
  repro_issue_361_int8_blob();
  fprintf(stderr, "-> running #334...\n");
  repro_issue_334_l2_scale();
  fprintf(stderr, "-> running #362/#363...\n");
  repro_issue_362_363_arith();
  printf("test_upstream_repro: ALL 4 UPSTREAM BUG CLASSES REPRODUCED\n");
  return 0;
}
