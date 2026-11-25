#include "test_common.h"
#include "sqlite3.h"
#include "sve_hybrid/sve_hybrid_rank.h"
#include "sve_hybrid/sve_sqlite_ext.h"
#include "sve_hybrid/sve_vtab_guard.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void exec_ok(sqlite3 *db, const char *sql) {
  char *err = NULL;
  int rc = sqlite3_exec(db, sql, NULL, NULL, &err);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "SQL failed (%d): %s -> %s\n", rc, sql,
            err ? err : "(null)");
    sqlite3_free(err);
    abort();
  }
}

static void test_hybrid_fusion(void) {
  sqlite3 *db = NULL;
  SVE_ASSERT_EQ_INT(sqlite3_open(":memory:", &db), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sve_hybrid_register(db, NULL), SQLITE_OK);

  exec_ok(db,
          "CREATE VIRTUAL TABLE kb USING sve0("
          "  emb float[4] index=hnsw metric=cosine,"
          "  cat text,"
          "  body text fulltext,"
          "  redact=1"
          ");");

  /* Doc 1 has closest dense vector AND exact thermal runaway keyword */
  exec_ok(db,
          "INSERT INTO kb(rowid, emb, cat, body) VALUES "
          "(1, '[0.95, 0.05, 0.0, 0.0]', 'battery', "
          "'Lithium cell thermal runaway mitigation with imsi=460001234567890');");
  /* Doc 2 has close vector but unrelated text */
  exec_ok(db,
          "INSERT INTO kb(rowid, emb, cat, body) VALUES "
          "(2, '[0.99, 0.01, 0.0, 0.0]', 'battery', "
          "'General DC bus voltage telemetry summary');");
  /* Doc 3 has distinct vector but lexical mention */
  exec_ok(db,
          "INSERT INTO kb(rowid, emb, cat, body) VALUES "
          "(3, '[0.20, 0.80, 0.0, 0.0]', 'rail', "
          "'Thermal management around braking resistors');");

  sqlite3_stmt *st = NULL;
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db,
          "SELECT rowid, body, rrf_score FROM kb "
          "WHERE emb MATCH '[1.0, 0.0, 0.0, 0.0]' "
          "  AND text_query MATCH 'thermal runaway mitigation' "
          "  AND k = 3;",
          -1, &st, NULL),
      SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_ROW);
  /* Doc 1 ranks #1 under RRF because both dense vector AND BM25 match strongly */
  SVE_ASSERT_EQ_INT(sqlite3_column_int64(st, 0), 1);
  const char *body = (const char *)sqlite3_column_text(st, 1);
  SVE_ASSERT(strstr(body, "[REDACTED]") != NULL);
  SVE_ASSERT(strstr(body, "460001234567890") == NULL);
  SVE_ASSERT(sqlite3_column_double(st, 2) > 0.0);
  sqlite3_finalize(st);

  /* Convex fusion alpha=0.35 verification */
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db,
          "SELECT rowid, rrf_score FROM kb "
          "WHERE emb MATCH '[1.0, 0.0, 0.0, 0.0]' "
          "  AND text_query MATCH 'thermal runaway mitigation' "
          "  AND hybrid_alpha = 0.35 AND k = 2;",
          -1, &st, NULL),
      SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_ROW);
  SVE_ASSERT_EQ_INT(sqlite3_column_int64(st, 0), 1);
  sqlite3_finalize(st);
  sqlite3_close(db);
}

static void test_fts_sanitizer(void) {
  char out[512];
  SVE_ASSERT_EQ_INT(
      sve_fts5_sanitize_query("C++ OR (NOT \"thermal-runaway\": 100%!)", 16U,
                              out, sizeof(out)),
      SVE_OK);
  /* Should quote FTS5 keywords cleanly without SQL syntax failure */
  sqlite3 *db = NULL;
  SVE_ASSERT_EQ_INT(sqlite3_open(":memory:", &db), SQLITE_OK);
  exec_ok(db, "CREATE VIRTUAL TABLE f USING fts5(txt, tokenize='unicode61');");
  exec_ok(db, "INSERT INTO f(rowid, txt) VALUES (1, 'thermal runaway C test');");
  sqlite3_stmt *st = NULL;
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(db, "SELECT rowid FROM f WHERE f MATCH ?1;", -1, &st,
                         NULL),
      SQLITE_OK);
  sqlite3_bind_text(st, 1, out, -1, SQLITE_STATIC);
  SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_ROW);
  SVE_ASSERT_EQ_INT(sqlite3_column_int64(st, 0), 1);
  sqlite3_finalize(st);
  sqlite3_close(db);
}

static void test_privacy_15_15(void) {
  static const char *const k_secrets[15] = {
      "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.sig99",
      "imsi=460001234567890 registered",
      "imei: 867530912345678",
      "msisdn=+86" "13800" "138000",
      "mobile: 13912" "345678",
      "api_key=\"live_token_9876543210abc\"",
      "sk-" "proj-abcDEF1234567890xyz",
      "ghp" "_AbCdEfGhIjKlMnOpQrSt1234",
      "AKIA" "IOSFODNN7EXAMPLE",
      "postgresql+asyncpg://svc:SuperSecret99@db.internal:5432/vec",
      "quic_ticket=qt_a1b2c3d4e5f6g7h8i9j0",
      "password = \"trickyPass!8899\"",
      "-----" "BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqh\n-----" "END PRIVATE KEY-----",
      "+86-138-1234-5678",
      "auth_token: tok_edge_991122334455"};

  static const char *const k_benign[15] = {
      "UUID 550e8400-e29b-41d4-a716-446655440000 verified",
      "SHA256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "ISO date 2026-11-25T12:00:00Z",
      "order_id=13800" "138000",
      "Bearer authentication specification RFC 6750",
      "latency p99 = 0.42 ms across 10000 vectors",
      "SQLite WAL checkpoint sector 12",
      "embedding dimension D=384",
      "tx_hash=13912" "345678 block 90012",
      "HNSW M=16 ef_construction=100",
      "temperature sensor 23.45 C",
      "worker-03 idle",
      "compression ratio 3.04x",
      "batch 13700" "137000 checksum valid",
      "HTTP 200 OK elapsed 1 ms"};

  char buf[4096];
  for (size_t i = 0U; i < 15U; i++) {
    size_t nr = 0U;
    SVE_ASSERT_EQ_INT(
        sve_privacy_scrub(k_secrets[i], strlen(k_secrets[i]), buf, sizeof(buf),
                          &nr),
        SVE_OK);
    SVE_ASSERT(nr > 0U);
    SVE_ASSERT(strstr(buf, "[REDACTED]") != NULL);
  }
  for (size_t i = 0U; i < 15U; i++) {
    size_t nr = 0U;
    SVE_ASSERT_EQ_INT(
        sve_privacy_scrub(k_benign[i], strlen(k_benign[i]), buf, sizeof(buf),
                          &nr),
        SVE_OK);
    SVE_ASSERT_EQ_INT(nr, 0U);
    SVE_ASSERT(strcmp(buf, k_benign[i]) == 0);
  }
}

static void test_loadable_dll(const char *dll_path) {
  sqlite3 *db = NULL;
  SVE_ASSERT_EQ_INT(sqlite3_open(":memory:", &db), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_enable_load_extension(db, 1), SQLITE_OK);
  char *err = NULL;
  int rc = sqlite3_load_extension(db, dll_path, "sqlite3_svehybrid_init", &err);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "load_extension failed for %s: %s\n", dll_path,
            err ? err : "(null)");
    sqlite3_free(err);
    abort();
  }
  sqlite3_stmt *st = NULL;
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(db, "SELECT sve_version();", -1, &st, NULL),
      SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_ROW);
  SVE_ASSERT(strstr((const char *)sqlite3_column_text(st, 0), "sve-hybrid") !=
             NULL);
  sqlite3_finalize(st);

  exec_ok(db, "CREATE VIRTUAL TABLE d USING sve0(v float[2] index=hnsw);");
  exec_ok(db, "INSERT INTO d(rowid, v) VALUES (1, '[1.0, 0.0]');");
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db, "SELECT rowid FROM d WHERE v MATCH '[1.0, 0.0]' AND k = 1;", -1,
          &st, NULL),
      SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_ROW);
  SVE_ASSERT_EQ_INT(sqlite3_column_int64(st, 0), 1);
  sqlite3_finalize(st);
  sqlite3_close(db);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    test_hybrid_fusion();
    test_fts_sanitizer();
    test_privacy_15_15();
    return 0;
  }
  if (strcmp(argv[1], "fusion") == 0) {
    test_hybrid_fusion();
  } else if (strcmp(argv[1], "fts_sanitize") == 0) {
    test_fts_sanitizer();
  } else if (strcmp(argv[1], "privacy") == 0) {
    test_privacy_15_15();
  } else {
    test_loadable_dll(argv[1]);
  }
  return 0;
}
