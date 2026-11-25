#include "test_common.h"
#include "sqlite3.h"
#include "sve_hybrid/sve_sqlite_ext.h"

#include <stdint.h>
#include <stdio.h>

static void exec_sql(sqlite3 *db, const char *sql) {
  char *err = NULL;
  int rc = sqlite3_exec(db, sql, NULL, NULL, &err);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "SQL failed (%d): %s -> %s\n", rc, sql,
            err ? err : "(null)");
    sqlite3_free(err);
    abort();
  }
}

/* #333 fix: failed explicit-txn INSERT leaves 0 ghost rows */
static void test_ghost_row_fix(void) {
  sqlite3 *db = NULL;
  SVE_ASSERT_EQ_INT(sqlite3_open(":memory:", &db), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sve_hybrid_register(db, NULL), SQLITE_OK);

  exec_sql(db,
           "CREATE VIRTUAL TABLE docs USING sve0("
           "  emb float[4] index=hnsw,"
           "  +code integer"
           ");");
  exec_sql(db,
           "INSERT INTO docs(rowid, emb, code) VALUES "
           "(1, '[0.0, 1.0, 0.0, 0.0]', 100);");

  exec_sql(db, "BEGIN;");
  char *err = NULL;
  int rc = sqlite3_exec(
      db,
      "INSERT INTO docs(rowid, emb, code) VALUES "
      "(42, '[1.0, 0.0, 0.0, 0.0]', 'invalid_int_text');",
      NULL, NULL, &err);
  SVE_ASSERT_EQ_INT(rc, SQLITE_CONSTRAINT);
  sqlite3_free(err);
  exec_sql(db, "COMMIT;");

  sqlite3_stmt *st = NULL;
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db,
          "SELECT rowid, code FROM docs "
          "WHERE emb MATCH '[1.0, 0.0, 0.0, 0.0]' AND k = 5;",
          -1, &st, NULL),
      SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_ROW);
  SVE_ASSERT_EQ_INT(sqlite3_column_int64(st, 0), 1);
  SVE_ASSERT_EQ_INT(sqlite3_column_int64(st, 1), 100);
  /* Zero second rows -> rowid 42 never leaked */
  SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_DONE);
  sqlite3_finalize(st);
  sqlite3_close(db);
}

/* #361 fix: untyped raw bound BLOB accepted cleanly on int8 & float columns */
static void test_bound_int8_blob(void) {
  sqlite3 *db = NULL;
  SVE_ASSERT_EQ_INT(sqlite3_open(":memory:", &db), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sve_hybrid_register(db, NULL), SQLITE_OK);
  exec_sql(db,
           "CREATE VIRTUAL TABLE ti USING sve0("
           "  v int8[4] index=hnsw metric=l2,"
           "  cat text"
           ");");

  int8_t v1[4] = {10, -20, 30, -40};
  int8_t v2[4] = {-50, 60, -70, 80};
  sqlite3_stmt *ins = NULL;
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db, "INSERT INTO ti(rowid, v, cat) VALUES (?1, ?2, ?3);", -1, &ins,
          NULL),
      SQLITE_OK);
  sqlite3_bind_int64(ins, 1, 7);
  sqlite3_bind_blob(ins, 2, v1, 4, SQLITE_STATIC);
  sqlite3_bind_text(ins, 3, "sensor", -1, SQLITE_STATIC);
  SVE_ASSERT_EQ_INT(sqlite3_step(ins), SQLITE_DONE);

  sqlite3_reset(ins);
  sqlite3_bind_int64(ins, 1, 8);
  sqlite3_bind_blob(ins, 2, v2, 4, SQLITE_STATIC);
  sqlite3_bind_text(ins, 3, "noise", -1, SQLITE_STATIC);
  SVE_ASSERT_EQ_INT(sqlite3_step(ins), SQLITE_DONE);
  sqlite3_finalize(ins);

  sqlite3_stmt *q = NULL;
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db, "SELECT rowid, cat, distance FROM ti WHERE v MATCH ?1 AND k = 1;",
          -1, &q, NULL),
      SQLITE_OK);
  sqlite3_bind_blob(q, 1, v1, 4, SQLITE_STATIC);
  SVE_ASSERT_EQ_INT(sqlite3_step(q), SQLITE_ROW);
  SVE_ASSERT_EQ_INT(sqlite3_column_int64(q, 0), 7);
  SVE_ASSERT(strcmp((const char *)sqlite3_column_text(q, 1), "sensor") == 0);
  SVE_ASSERT_NEAR(sqlite3_column_double(q, 2), 0.0, 1e-9);
  sqlite3_finalize(q);

  sqlite3_close(db);
}

static void test_crud_replace_delete(void) {
  sqlite3 *db = NULL;
  SVE_ASSERT_EQ_INT(sqlite3_open(":memory:", &db), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sve_hybrid_register(db, NULL), SQLITE_OK);

  exec_sql(db, "CREATE VIRTUAL TABLE tc USING sve0(emb float[3] index=hnsw);");
  exec_sql(db, "INSERT INTO tc(rowid, emb) VALUES (1, '[1,0,0]');");
  exec_sql(db, "INSERT INTO tc(rowid, emb) VALUES (2, '[0,1,0]');");
  exec_sql(db, "INSERT OR REPLACE INTO tc(rowid, emb) VALUES (1, '[0,0,1]');");
  exec_sql(db, "DELETE FROM tc WHERE rowid = 2;");

  sqlite3_stmt *st = NULL;
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db,
          "SELECT rowid, distance FROM tc WHERE emb MATCH '[0,0,1]' AND k = 5;",
          -1, &st, NULL),
      SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_ROW);
  SVE_ASSERT_EQ_INT(sqlite3_column_int64(st, 0), 1);
  SVE_ASSERT_NEAR(sqlite3_column_double(st, 1), 0.0, 1e-9);
  SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_DONE);
  sqlite3_finalize(st);
  sqlite3_close(db);
}

static void test_rollback_safety(void) {
  sqlite3 *db = NULL;
  SVE_ASSERT_EQ_INT(sqlite3_open(":memory:", &db), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sve_hybrid_register(db, NULL), SQLITE_OK);

  exec_sql(db, "CREATE VIRTUAL TABLE tr USING sve0(emb float[3] index=hnsw);");
  exec_sql(db, "INSERT INTO tr(rowid, emb) VALUES (10, '[1,0,0]');");
  exec_sql(db, "BEGIN;");
  exec_sql(db, "INSERT INTO tr(rowid, emb) VALUES (20, '[0,1,0]');");
  exec_sql(db, "ROLLBACK;");

  sqlite3_stmt *st = NULL;
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db, "SELECT rowid FROM tr WHERE emb MATCH '[0,1,0]' AND k = 5;", -1,
          &st, NULL),
      SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_ROW);
  SVE_ASSERT_EQ_INT(sqlite3_column_int64(st, 0), 10);
  SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_DONE);
  sqlite3_finalize(st);
  sqlite3_close(db);
}

static void test_disk_persistence(void) {
  const char *db_path = "test_persist_sve0.db";
  remove(db_path);

  sqlite3 *db = NULL;
  SVE_ASSERT_EQ_INT(sqlite3_open(db_path, &db), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sve_hybrid_register(db, NULL), SQLITE_OK);
  exec_sql(db,
           "CREATE VIRTUAL TABLE dp USING sve0("
           "  emb float[3] index=hnsw metric=cosine,"
           "  tag text"
           ");");
  exec_sql(db, "INSERT INTO dp(rowid, emb, tag) VALUES (5, '[1,0,0]', 'ok');");
  exec_sql(db, "INSERT INTO dp(rowid, emb, tag) VALUES (6, '[0,1,0]', 'alt');");
  sqlite3_close(db);

  SVE_ASSERT_EQ_INT(sqlite3_open(db_path, &db), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sve_hybrid_register(db, NULL), SQLITE_OK);
  sqlite3_stmt *st = NULL;
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db,
          "SELECT rowid, tag, distance FROM dp WHERE emb MATCH '[1,0,0]' AND "
          "k = 1;",
          -1, &st, NULL),
      SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_ROW);
  SVE_ASSERT_EQ_INT(sqlite3_column_int64(st, 0), 5);
  SVE_ASSERT(strcmp((const char *)sqlite3_column_text(st, 1), "ok") == 0);
  SVE_ASSERT_NEAR(sqlite3_column_double(st, 2), 0.0, 1e-6);
  sqlite3_finalize(st);
  sqlite3_close(db);
  remove(db_path);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    test_ghost_row_fix();
    test_bound_int8_blob();
    test_crud_replace_delete();
    test_rollback_safety();
    test_disk_persistence();
    return 0;
  }
  if (strcmp(argv[1], "ghost_row") == 0) {
    test_ghost_row_fix();
  } else if (strcmp(argv[1], "bound_int8") == 0) {
    test_bound_int8_blob();
  } else if (strcmp(argv[1], "crud") == 0) {
    test_crud_replace_delete();
  } else if (strcmp(argv[1], "rollback") == 0) {
    test_rollback_safety();
  } else if (strcmp(argv[1], "disk_persist") == 0) {
    test_disk_persistence();
  } else {
    return 1;
  }
  return 0;
}
