#include "test_common.h"
#include "sve_hybrid/sve_hnsw.h"
#include "sve_hybrid/sve_hybrid_rank.h"
#include "sve_hybrid/sve_numeric.h"
#include "sve_hybrid/sve_sqlite_ext.h"
#include "sve_hybrid/sve_vtab_guard.h"

#include <float.h>
#include <locale.h>

static void exec_ok(sqlite3 *db, const char *sql) {
  char *err = NULL;
  int rc = sqlite3_exec(db, sql, NULL, NULL, &err);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "SQL failure (%d): %s\n", rc, err ? err : "");
  }
  sqlite3_free(err);
  SVE_ASSERT_EQ_INT(rc, SQLITE_OK);
}

static sqlite3 *open_db(const char *path) {
  sqlite3 *db = NULL;
  SVE_ASSERT_EQ_INT(sqlite3_open(path, &db), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sve_hybrid_register(db, NULL), SQLITE_OK);
  return db;
}

static sqlite3_stmt *query(sqlite3 *db, const char *sql) {
  sqlite3_stmt *st = NULL;
  SVE_ASSERT_EQ_INT(sqlite3_prepare_v2(db, sql, -1, &st, NULL), SQLITE_OK);
  return st;
}

static int64_t single_int(sqlite3 *db, const char *sql) {
  sqlite3_stmt *st = query(db, sql);
  SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_ROW);
  int64_t result = sqlite3_column_int64(st, 0);
  SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_DONE);
  SVE_ASSERT_EQ_INT(sqlite3_finalize(st), SQLITE_OK);
  return result;
}

static void test_json_locale_extremes(void) {
  static const char *const locales[] = {
      "de-DE", "German_Germany.1252", "German", "de_DE.UTF-8", "fr_FR.UTF-8"};
  bool comma = false;
  for (size_t i = 0U; i < sizeof(locales) / sizeof(locales[0]); i++) {
    if (setlocale(LC_NUMERIC, locales[i]) &&
        strcmp(localeconv()->decimal_point, ",") == 0) {
      comma = true;
      break;
    }
  }
#ifdef _WIN32
  SVE_ASSERT(comma);
#endif
  puts(comma ? "Verified with a comma decimal locale" : "Using the C locale");

  const char slice[] = {'[', '1', '.', '2', '5', ',', '2', ']'};
  float *f32 = NULL;
  size_t dims = 0U;
  SVE_ASSERT_EQ_INT(sve_parse_f32_json(slice, sizeof(slice), 2U, &f32, &dims),
                    SVE_OK);
  SVE_ASSERT_EQ_INT(dims, 2);
  SVE_ASSERT_NEAR(f32[0], 1.25, 0.0);
  SVE_ASSERT_NEAR(f32[1], 2.0, 0.0);
  free(f32);
  const char bounds[] = "[3.40282347e38,1e-44]";
  SVE_ASSERT_EQ_INT(sve_parse_f32_json(bounds, strlen(bounds), 2U, &f32, &dims),
                    SVE_OK);
  SVE_ASSERT(f32[0] == FLT_MAX && f32[1] > 0.0f && f32[1] < FLT_MIN);
  free(f32);
  const char *const invalid_f32[] = {
      "[.5]", "[1.]", "[01]", "[-]", "[0x2]", "[3.5e38]", "[1e]", "[1\v]"};
  for (size_t i = 0U; i < sizeof(invalid_f32) / sizeof(invalid_f32[0]); i++) {
    SVE_ASSERT_EQ_INT(sve_parse_f32_json(invalid_f32[i], strlen(invalid_f32[i]),
                                        2U, &f32, &dims), SVE_ERR_INVALID);
    SVE_ASSERT(f32 == NULL && dims == 0U);
  }
  const char int_slice[] = {'[', '-', '1', '2', '8', ',', '1', '2', '7', ']'};
  int8_t *i8 = NULL;
  SVE_ASSERT_EQ_INT(sve_parse_i8_json(int_slice, sizeof(int_slice), 2U, &i8, &dims),
                    SVE_OK);
  SVE_ASSERT(dims == 2U && i8[0] == -128 && i8[1] == 127);
  free(i8);
  const char *const invalid_i8[] = {"[12abc]", "[-]", "[00]", "[128]", "[-129]",
                                    "[1.0]", "[1e0]"};
  for (size_t i = 0U; i < sizeof(invalid_i8) / sizeof(invalid_i8[0]); i++) {
    SVE_ASSERT_EQ_INT(sve_parse_i8_json(invalid_i8[i], strlen(invalid_i8[i]),
                                       2U, &i8, &dims), SVE_ERR_INVALID);
    SVE_ASSERT(i8 == NULL && dims == 0U);
  }
  SVE_ASSERT(setlocale(LC_NUMERIC, "C") != NULL);

  const float large[] = {3.0e38f, 3.0e38f};
  const float opposite[] = {-3.0e38f, -3.0e38f};
  const float tiny[] = {1.0e-44f, 0.0f};
  const float zero[] = {0.0f, 0.0f};
  double distance = 0.0;
  SVE_ASSERT_EQ_INT(sve_distance_l2_f32(large, opposite, 2U, &distance), SVE_OK);
  SVE_ASSERT(isfinite(distance) && distance > (double)FLT_MAX);
  SVE_ASSERT_EQ_INT(sve_distance_l2_f32(tiny, zero, 2U, &distance), SVE_OK);
  SVE_ASSERT_NEAR(distance, (double)tiny[0], 0.0);
  SVE_ASSERT_EQ_INT(sve_distance_cosine_f32(large, opposite, 2U, &distance), SVE_OK);
  SVE_ASSERT_NEAR(distance, 2.0, 1e-15);
  SVE_ASSERT_EQ_INT(sve_distance_cosine_f32(tiny, tiny, 2U, &distance), SVE_OK);
  SVE_ASSERT_NEAR(distance, 0.0, 1e-15);
  SVE_ASSERT_EQ_INT(sve_distance_cosine_f32(tiny, large, 2U, &distance), SVE_OK);
  SVE_ASSERT_NEAR(distance, 1.0 - sqrt(0.5), 1e-15);

  int8_t quant[2];
  float scale = 0.0f, back[2];
  SVE_ASSERT_EQ_INT(sve_quantize_f32_to_sq8(tiny, 2U, quant, &scale), SVE_OK);
  SVE_ASSERT(scale > 0.0f);
  SVE_ASSERT_EQ_INT(sve_dequantize_sq8_to_f32(quant, scale, 2U, back), SVE_OK);
  SVE_ASSERT(back[0] > 0.0f && back[1] == 0.0f);
  const float top[] = {FLT_MAX};
  SVE_ASSERT_EQ_INT(sve_quantize_f32_to_sq8(top, 1U, quant, &scale), SVE_OK);
  SVE_ASSERT_EQ_INT(sve_dequantize_sq8_to_f32(quant, scale, 1U, back), SVE_OK);
  SVE_ASSERT(isfinite(back[0]));
  SVE_ASSERT_EQ_INT(sve_dequantize_sq8_to_f32(quant, FLT_MAX, 1U, back),
                    SVE_ERR_OVERFLOW);
}

static void test_rank_and_sanitize(void) {
  const sve_search_hit_t vec[] = {{5, -DBL_MAX}, {7, DBL_MAX}};
  const sve_search_hit_t bm[] = {{5, -DBL_MAX}, {7, DBL_MAX}};
  sve_hybrid_hit_t out[2];
  size_t n = 0U;
  SVE_ASSERT_EQ_INT(sve_fuse_convex(vec, 2U, bm, 2U, 0.5, 2U, out, &n), SVE_OK);
  SVE_ASSERT_EQ_INT(n, 2U);
  SVE_ASSERT_EQ_INT(out[0].rowid, 5);
  SVE_ASSERT_NEAR(out[0].fused_score, 1.0, 0.0);
  SVE_ASSERT_NEAR(out[1].fused_score, 0.0, 0.0);

  char text[64];
  SVE_ASSERT_EQ_INT(sve_fts5_sanitize_query("?" "?!!++", 32U, text, sizeof(text)), SVE_OK);
  SVE_ASSERT_EQ_INT(text[0], 0);
  SVE_ASSERT_EQ_INT(sve_fts5_sanitize_query("NOT OR NEAR AND", 32U, text, sizeof(text)),
                    SVE_OK);
  SVE_ASSERT(strcmp(text, "\"NOT\" OR \"OR\" OR \"NEAR\" OR \"AND\"") == 0);
  const char utf8[] = "\xE6\xB5\x8B" " \xF0\x9F\x98\x80";
  SVE_ASSERT_EQ_INT(sve_fts5_sanitize_query(utf8, 32U, text, 11U), SVE_OK);
  SVE_ASSERT(strcmp(text, "\"\xE6\xB5\x8B\"") == 0);
  SVE_ASSERT_EQ_INT(sve_fts5_sanitize_query("\xE6\xB5", 32U, text, sizeof(text)),
                    SVE_ERR_INVALID);
  SVE_ASSERT_EQ_INT(text[0], 0);
}

static void check_scrub(const char *input, const char *expected, size_t expected_count) {
  char out[512];
  size_t count = 0U;
  SVE_ASSERT_EQ_INT(sve_privacy_scrub(input, strlen(input), out, sizeof(out), &count),
                    SVE_OK);
  SVE_ASSERT_EQ_INT(count, expected_count);
  SVE_ASSERT(strcmp(out, expected) == 0);
}

static void test_privacy_boundaries(void) {
  check_scrub("password=\"dummy ! value\";sk-" "ABCD1234unitTESTtag.ghp" "_ABCD1234unitTESTtag",
              "password=\"[REDACTED]\";[REDACTED].[REDACTED]", 3U);
  check_scrub("-----" "BEGIN EC " "PRIVATE KEY-----\nfixture\n-----" "END " "PRIVATE KEY-----\n"
              "still-private\n-----" "END EC " "PRIVATE KEY-----; password='dummy'",
              "[REDACTED]; password='[REDACTED]'", 2U);
  const char decoys[] = "ip 192.0.2.10, build v2.16.17, unix 1761234567, "
                        "2026-11-25T12:01:02Z, trace_id=13912" "345678";
  check_scrub(decoys, decoys, 0U);
  size_t bytes = SVE_MAX_TEXT_BYTES;
  char *input = (char *)malloc(bytes + 1U);
  char *output = (char *)malloc(bytes + 1U);
  SVE_ASSERT(input && output);
  memset(input, 'a', bytes);
  input[bytes] = '\0';
  size_t count = 111U;
  SVE_ASSERT_EQ_INT(sve_privacy_scrub(input, bytes, output, bytes + 1U, &count), SVE_OK);
  SVE_ASSERT_EQ_INT(count, 0U);
  SVE_ASSERT(strcmp(input, output) == 0);
  const char pem[] = "-----" "BEGIN OPENSSH " "PRIVATE KEY-----\n";
  memcpy(input, pem, sizeof(pem) - 1U);
  SVE_ASSERT_EQ_INT(sve_privacy_scrub(input, bytes, output, bytes + 1U, &count), SVE_OK);
  SVE_ASSERT_EQ_INT(count, 1U);
  SVE_ASSERT(strcmp(output, "[REDACTED]") == 0);
  SVE_ASSERT_EQ_INT(sve_privacy_scrub(input, bytes, output, 5U, &count),
                    SVE_ERR_LIMIT);
  SVE_ASSERT_EQ_INT(output[0], 0);
  memset(input, 'u', 160U);
  memcpy(input, "test://", 7U);
  const char suffix[] = ":fixture-password@host.example/";
  memcpy(input + 160U, suffix, sizeof(suffix));
  SVE_ASSERT_EQ_INT(sve_privacy_scrub(input, strlen(input), output, bytes + 1U, &count),
                    SVE_OK);
  SVE_ASSERT_EQ_INT(count, 1U);
  SVE_ASSERT(strcmp(output + 160U, ":[REDACTED]@host.example/") == 0);
  free(input);
  free(output);
}

static void test_large_filtered_plan(void) {
  sqlite3 *db = open_db(":memory:");
  exec_ok(db, "CREATE VIRTUAL TABLE v USING sve0(e float[1], rankval integer, "
              "tag text, body text fulltext);"
              "INSERT INTO v(rowid,e,rankval,tag,body) VALUES"
              "(1,'[0]',0,'Other','alpha'),(2,'[1]',1,'Tag','alpha');");
  sqlite3_str *builder = sqlite3_str_new(db);
  sqlite3_str_appendall(builder, "SELECT rowid FROM v WHERE e MATCH '[0]' AND k=1");
  for (int i = 0; i < 45; i++) {
    sqlite3_str_appendf(builder, " AND rankval > %d", -1 - i);
  }
  sqlite3_str_appendall(builder, " AND rankval > 0 AND tag COLLATE NOCASE='tag'"
                                 " AND text_query MATCH 'alpha' AND hybrid_alpha=0.25;");
  char *sql = sqlite3_str_finish(builder);
  SVE_ASSERT(sql);
  SVE_ASSERT_EQ_INT(single_int(db, sql), 2);
  sqlite3_free(sql);

  exec_ok(db, "CREATE VIRTUAL TABLE f USING sve0(e float[1] index=flat, "
              "bucket integer, body text fulltext);"
              "WITH RECURSIVE ids(i) AS (VALUES(1) UNION ALL SELECT i+1 FROM ids WHERE i<45) "
              "INSERT INTO f(rowid,e,bucket,body) SELECT i,'[0]',0,'alpha alpha alpha' FROM ids;"
              "INSERT INTO f(rowid,e,bucket,body) VALUES"
              "(99,'[0]',1,'unrelated'),(100,'[1]',1,'alpha');");
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT rowid FROM f WHERE e MATCH '[0]' AND k=1 "
                                  "AND text_query MATCH 'alpha' AND hybrid_alpha=0 AND bucket=1;"),
                    100);
  SVE_ASSERT_EQ_INT(sqlite3_close(db), SQLITE_OK);
}

static void test_query_controls(void) {
  sqlite3 *db = open_db(":memory:");
  exec_ok(db, "CREATE VIRTUAL TABLE v USING sve0(e float[1], body text fulltext);"
              "INSERT INTO v(rowid,e,body) VALUES(1,'[0]','sample');");
  SVE_ASSERT_EQ_INT(sqlite3_exec(db, "SELECT rowid FROM v WHERE e MATCH '[0]' "
                                    "AND e MATCH '[1]';", NULL, NULL, NULL),
                    SQLITE_ERROR);
  sqlite3_stmt *st = query(db, "SELECT rowid FROM v WHERE e MATCH '[0]' AND k=?1 "
                               "AND text_query MATCH ?2 AND hybrid_alpha=?3;");
  for (int i = 0; i < 9; i++) {
    SVE_ASSERT_EQ_INT(sqlite3_bind_int64(st, 1,
        i == 1 ? -1 : (i == 2 ? INT64_MAX : 1)), SQLITE_OK);
    if (i == 3) {
      SVE_ASSERT_EQ_INT(sqlite3_bind_null(st, 2), SQLITE_OK);
    } else {
      SVE_ASSERT_EQ_INT(sqlite3_bind_text(st, 2, "sample", -1, SQLITE_STATIC), SQLITE_OK);
    }
    SVE_ASSERT_EQ_INT(sqlite3_bind_double(st, 3, i == 4 ? NAN : (i == 5 ? INFINITY : 0.25)),
                      SQLITE_OK);
    if (i == 1 || i == 2 || i == 4 || i == 5) {
      SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_ERROR);
      SVE_ASSERT_EQ_INT(sqlite3_reset(st), SQLITE_ERROR);
    } else {
      if (i != 3) {
        SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_ROW);
        SVE_ASSERT_EQ_INT(sqlite3_column_int64(st, 0), 1);
      }
      SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_DONE);
      SVE_ASSERT_EQ_INT(sqlite3_reset(st), SQLITE_OK);
    }
  }
  sqlite3_finalize(st);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT count(*) FROM v WHERE e MATCH '[0]' AND k=4096;"),
                    1);
  exec_ok(db, "UPDATE v SET body='edited' WHERE e MATCH '[0]' AND k=1 "
              "AND text_query MATCH 'sample' AND hybrid_alpha=0.25;");
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT rowid FROM v_fts WHERE v_fts MATCH 'edited';"), 1);
  SVE_ASSERT_EQ_INT(sqlite3_close(db), SQLITE_OK);
}

static int deny_insert(void *opaque, int action, const char *table,
                       const char *arg, const char *schema, const char *trigger) {
  (void)arg;
  (void)schema;
  (void)trigger;
  return action == SQLITE_INSERT && table && strcmp(table, (const char *)opaque) == 0
             ? SQLITE_DENY : SQLITE_OK;
}

static void test_shadow_failure(void) {
  const char *path = "sve_audit_shadow.db";
  remove(path);
  sqlite3 *db = open_db(path);
  exec_ok(db, "CREATE VIRTUAL TABLE v USING sve0(e float[1], body text fulltext);"
              "INSERT INTO v(rowid,e,body) VALUES(1,'[1]','original');"
              "BEGIN; INSERT INTO v(rowid,e,body) VALUES(2,'[2]','earlier');");
  SVE_ASSERT_EQ_INT(sqlite3_set_authorizer(db, deny_insert, "v_fts"), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_exec(db, "INSERT OR REPLACE INTO v(rowid,e,body) "
                                    "VALUES(1,'[9]','fresh');", NULL, NULL, NULL),
                    SQLITE_AUTH);
  SVE_ASSERT_EQ_INT(sqlite3_exec(db, "UPDATE v SET e='[9]', body='fresh' WHERE rowid=1;",
                                NULL, NULL, NULL), SQLITE_AUTH);
  SVE_ASSERT_EQ_INT(sqlite3_set_authorizer(db, NULL, NULL), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_get_autocommit(db), 0);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT count(*) FROM v_rows WHERE "
                                  "(id=1 AND c0='original') OR (id=2 AND c0='earlier');"), 2);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT rowid FROM v_fts WHERE v_fts MATCH 'original';"), 1);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT rowid FROM v WHERE e MATCH '[1]' AND k=1;"), 1);
  exec_ok(db, "COMMIT; BEGIN; INSERT INTO v(rowid,e,body) VALUES(3,'[3]','pending');");
  int old_limit = sqlite3_limit(db, SQLITE_LIMIT_LENGTH, 125);
  SVE_ASSERT_EQ_INT(sqlite3_exec(db, "COMMIT;", NULL, NULL, NULL), SQLITE_TOOBIG);
  sqlite3_limit(db, SQLITE_LIMIT_LENGTH, old_limit);
  if (!sqlite3_get_autocommit(db)) {
    exec_ok(db, "ROLLBACK;");
  }
  SVE_ASSERT_EQ_INT(sqlite3_close(db), SQLITE_OK);
  db = open_db(path);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT count(*) FROM v_rows;"), 2);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT count(*) FROM v_fts;"), 2);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT count(*) FROM v WHERE e MATCH '[0]' AND k=10;"), 2);
  SVE_ASSERT_EQ_INT(sqlite3_close(db), SQLITE_OK);
  remove(path);
}

static void test_savepoint_and_conflicts(void) {
  const char *path = "sve_audit_savepoint.db";
  remove(path);
  sqlite3 *db = open_db(path);
  exec_ok(db, "CREATE VIRTUAL TABLE v USING sve0(e float[1], body text fulltext);"
              "BEGIN; INSERT INTO v(rowid,e,body) VALUES(1,'[1]','one');"
              "SAVEPOINT sp; INSERT INTO v(rowid,e,body) VALUES(2,'[2]','two');"
              "ROLLBACK TO sp; RELEASE sp; COMMIT;");
  SVE_ASSERT_EQ_INT(sqlite3_close(db), SQLITE_OK);
  db = open_db(path);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT rowid FROM v WHERE e MATCH '[1]' AND k=5;"), 1);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT count(*) FROM v_fts;"), 1);

  exec_ok(db, "BEGIN; INSERT INTO v(rowid,e,body) VALUES(3,'[3]','three');");
  SVE_ASSERT_EQ_INT(sqlite3_exec(db, "INSERT INTO v(rowid,e,body) VALUES"
                                    "(4,'[4]','four'),(1,'[6]','dupe');",
                                NULL, NULL, NULL), SQLITE_CONSTRAINT);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT count(*) FROM v_rows WHERE id=4;"), 0);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT count(*) FROM v WHERE e MATCH '[3]' AND k=10;"), 2);
  SVE_ASSERT_EQ_INT(sqlite3_exec(db, "INSERT OR FAIL INTO v(rowid,e,body) VALUES"
                                    "(5,'[5]','five'),(1,'[6]','dupe');",
                                NULL, NULL, NULL), SQLITE_CONSTRAINT);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT rowid FROM v WHERE e MATCH '[5]' AND k=1;"), 5);
  exec_ok(db, "INSERT OR IGNORE INTO v(rowid,e,body) VALUES(1,'[6]','dupe');");
  SVE_ASSERT_EQ_INT(sqlite3_changes(db), 0);
  exec_ok(db, "UPDATE OR REPLACE v SET rowid=3,e='[8]',body='moved' WHERE rowid=1;"
              "COMMIT;");
  SVE_ASSERT_EQ_INT(sqlite3_close(db), SQLITE_OK);
  db = open_db(path);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT rowid FROM v WHERE e MATCH '[8]' AND k=1;"), 3);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT rowid FROM v_fts WHERE v_fts MATCH 'moved';"), 3);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT count(*) FROM v_rows;"), 2);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT count(*) FROM v_fts;"), 2);
  SVE_ASSERT_EQ_INT(sqlite3_close(db), SQLITE_OK);
  remove(path);
}

static void test_connections(void) {
  const char *path = "sve_audit_wal.db";
  remove(path);
  sqlite3 *a = open_db(path);
  exec_ok(a, "PRAGMA journal_mode=WAL; CREATE VIRTUAL TABLE v USING sve0(e float[1]);"
             "INSERT INTO v(rowid,e) VALUES(1,'[1]');");
  sqlite3 *b = open_db(path);
  const char *near = "SELECT rowid FROM v WHERE e MATCH '[2]' AND k=1;";
  exec_ok(a, "BEGIN;");
  SVE_ASSERT_EQ_INT(single_int(a, near), 1);
  exec_ok(b, "INSERT INTO v(rowid,e) VALUES(2,'[2]');");
  SVE_ASSERT_EQ_INT(single_int(a, near), 1);
  exec_ok(a, "COMMIT;");
  SVE_ASSERT_EQ_INT(single_int(a, near), 2);
  exec_ok(a, "INSERT INTO v(rowid,e) VALUES(3,'[3]');");
  SVE_ASSERT_EQ_INT(single_int(b, "SELECT count(*) FROM v WHERE e MATCH '[2]' AND k=4;"), 3);
  SVE_ASSERT_EQ_INT(sqlite3_close(a), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_close(b), SQLITE_OK);
  remove(path);
}

static void test_identifiers(void) {
  sqlite3 *db = open_db(":memory:");
  exec_ok(db, "CREATE TABLE v_rows(id INTEGER); INSERT INTO v_rows VALUES(11);");
  SVE_ASSERT_EQ_INT(sqlite3_exec(db, "CREATE VIRTUAL TABLE v USING sve0(e float[1]);",
                                NULL, NULL, NULL), SQLITE_ERROR);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT id FROM v_rows;"), 11);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT count(*) FROM sqlite_schema WHERE name='v_info';"),
                    0);
  SVE_ASSERT_EQ_INT(sqlite3_exec(db, "CREATE VIRTUAL TABLE v2 USING sve0(e float[2] index=flatx);",
                                NULL, NULL, NULL), SQLITE_ERROR);
  SVE_ASSERT_EQ_INT(sqlite3_exec(db, "CREATE VIRTUAL TABLE v3 USING sve0(e bit[7]);",
                                NULL, NULL, NULL), SQLITE_ERROR);
  exec_ok(db, "ATTACH ':memory:' AS \"s\"\"e\";"
              "CREATE VIRTUAL TABLE \"s\"\"e\".\"d\"\"b\" "
              "USING sve0(\"e\" float[1], +\"body\" text fulltext);"
              "INSERT INTO \"s\"\"e\".\"d\"\"b\"(rowid,e,body) VALUES(7,'[7]','word');");
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT rowid FROM \"s\"\"e\".\"d\"\"b\" "
                                  "WHERE e MATCH '[7]' AND text_query MATCH 'word' AND k=1;"), 7);
  SVE_ASSERT_EQ_INT(sqlite3_exec(db, "ALTER TABLE \"s\"\"e\".\"d\"\"b\" RENAME TO moved;",
                                NULL, NULL, NULL), SQLITE_ERROR);
  exec_ok(db, "DROP TABLE \"s\"\"e\".\"d\"\"b\"; DETACH \"s\"\"e\";");
  SVE_ASSERT_EQ_INT(sqlite3_close(db), SQLITE_OK);
}

static void test_snapshot_connect(void) {
  const char *path = "sve_audit_snapshot.db";
  remove(path);
  sqlite3 *db = open_db(path);
  exec_ok(db, "CREATE VIRTUAL TABLE v USING sve0(e float[1]);"
              "CREATE VIRTUAL TABLE other USING sve0(e float[2]);"
              "INSERT INTO v(rowid,e) VALUES(1,'[1]');"
              "INSERT INTO other(rowid,e) VALUES(1,'[1,2]');");
  sqlite3_stmt *read = query(db, "SELECT val FROM v_info;");
  SVE_ASSERT_EQ_INT(sqlite3_step(read), SQLITE_ROW);
  int bytes = sqlite3_column_bytes(read, 0);
  SVE_ASSERT(bytes > 65);
  uint8_t *original = (uint8_t *)malloc((size_t)bytes);
  SVE_ASSERT(original);
  memcpy(original, sqlite3_column_blob(read, 0), (size_t)bytes);
  sqlite3_finalize(read);
  for (int mode = 0; mode < 3; mode++) {
    if (mode == 2) {
      exec_ok(db, "UPDATE v_info SET val=(SELECT val FROM other_info);");
    } else {
      sqlite3_stmt *write = query(db, "UPDATE v_info SET val=?1;");
      if (mode == 0) {
        original[bytes - 1] ^= 1U;
      }
      SVE_ASSERT_EQ_INT(sqlite3_bind_blob(write, 1, original, bytes - mode, SQLITE_TRANSIENT),
                        SQLITE_OK);
      SVE_ASSERT_EQ_INT(sqlite3_step(write), SQLITE_DONE);
      sqlite3_finalize(write);
      if (mode == 0) {
        original[bytes - 1] ^= 1U;
      }
    }
    SVE_ASSERT_EQ_INT(sqlite3_close(db), SQLITE_OK);
    db = open_db(path);
    sqlite3_stmt *st = NULL;
    SVE_ASSERT_EQ_INT(sqlite3_prepare_v2(db, "SELECT rowid FROM v WHERE e MATCH '[1]';",
                                       -1, &st, NULL), SQLITE_CORRUPT);
    sqlite3_finalize(st);
  }
  free(original);
  SVE_ASSERT_EQ_INT(sqlite3_close(db), SQLITE_OK);
  remove(path);
}

static void test_subtypes(const char *dll) {
  sqlite3 *db = NULL;
  SVE_ASSERT_EQ_INT(sqlite3_open(":memory:", &db), SQLITE_OK);
  sqlite3_stmt *active = query(db, "SELECT 1;");
  SVE_ASSERT_EQ_INT(sqlite3_step(active), SQLITE_ROW);
  SVE_ASSERT_EQ_INT(sve_hybrid_register(db, NULL), SQLITE_BUSY);
  sqlite3_finalize(active);
  if (dll) {
    SVE_ASSERT_EQ_INT(sqlite3_enable_load_extension(db, 1), SQLITE_OK);
    SVE_ASSERT_EQ_INT(sqlite3_load_extension(db, dll, NULL, NULL), SQLITE_OK);
  } else {
    SVE_ASSERT_EQ_INT(sve_hybrid_register(db, NULL), SQLITE_OK);
  }
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT sve_length(sve_int8('[1,2,3]'));"), 3);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT sve_length(sve_bit(x'10'));"), 8);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT sve_length(sve_add(sve_int8('[1]'),sve_int8('[2]')));"),
                    1);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT subtype(sve_int8('[1]'));"), SVE_ELEM_INT8);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT subtype(sve_f32('[1]'));"), SVE_ELEM_FLOAT32);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT subtype(sve_bit(x'10'));"), SVE_ELEM_BIT);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT count(*) FROM pragma_function_list "
                                  "WHERE name='sve_int8' AND flags & 1048576 != 0;"), 1);
  exec_ok(db, "CREATE VIRTUAL TABLE v USING sve0(e int8[3]);"
              "INSERT INTO v(e) VALUES(sve_int8('[1,2,3]'));"
              "CREATE TABLE data(raw BLOB); INSERT INTO data VALUES(x'010203');"
              "CREATE INDEX typed ON data(sve_length(sve_int8(raw)));");
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT sve_length(e) FROM v;"), 3);
  SVE_ASSERT_EQ_INT(single_int(db, "SELECT count(*) FROM data INDEXED BY typed "
                                  "WHERE sve_length(sve_int8(raw))=3;"), 1);
  SVE_ASSERT_EQ_INT(sqlite3_close(db), SQLITE_OK);
}

int main(int argc, char **argv) {
  SVE_ASSERT(argc > 1);
  const char *name = argv[1];
  if (strcmp(name, "json") == 0) {
    test_json_locale_extremes();
  } else if (strcmp(name, "rank") == 0) {
    test_rank_and_sanitize();
  } else if (strcmp(name, "privacy") == 0) {
    test_privacy_boundaries();
  } else if (strcmp(name, "planner") == 0) {
    test_large_filtered_plan();
  } else if (strcmp(name, "controls") == 0) {
    test_query_controls();
  } else if (strcmp(name, "shadow_failure") == 0) {
    test_shadow_failure();
  } else if (strcmp(name, "savepoint") == 0) {
    test_savepoint_and_conflicts();
  } else if (strcmp(name, "connections") == 0) {
    test_connections();
  } else if (strcmp(name, "identifiers") == 0) {
    test_identifiers();
  } else if (strcmp(name, "snapshot") == 0) {
    test_snapshot_connect();
  } else if (strcmp(name, "subtypes") == 0) {
    test_subtypes(argc > 2 ? argv[2] : NULL);
  } else {
    SVE_ASSERT(false);
  }
  return 0;
}
