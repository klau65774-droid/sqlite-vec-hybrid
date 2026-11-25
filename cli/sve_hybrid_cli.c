#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "sqlite-vec.h"
#include "sqlite3.h"
#include "sve_hybrid/sve_config.h"
#include "sve_hybrid/sve_hnsw.h"
#include "sve_hybrid/sve_hybrid_rank.h"
#include "sve_hybrid/sve_numeric.h"
#include "sve_hybrid/sve_sqlite_ext.h"
#include "sve_hybrid/sve_vtab_guard.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static double now_ms(void) {
  LARGE_INTEGER freq, cnt;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&cnt);
  return (double)cnt.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
#include <time.h>
static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
#endif

static int cmp_double_asc(const void *a, const void *b) {
  double va = *(const double *)a;
  double vb = *(const double *)b;
  if (va < vb) {
    return -1;
  }
  if (va > vb) {
    return 1;
  }
  return 0;
}

typedef struct latency {
  double p50;
  double p99;
} latency_t;

static latency_t latency_summary(double *samples, size_t n) {
  qsort(samples, n, sizeof(*samples), cmp_double_asc);
  return (latency_t){samples[(size_t)((double)(n - 1U) * 0.50)],
                      samples[(size_t)((double)(n - 1U) * 0.99)]};
}

static bool parse_u64(const char *text, uint64_t *out) {
  if (text[0] < '0' || text[0] > '9') {
    return false;
  }
  errno = 0;
  char *end = NULL;
  unsigned long long value = strtoull(text, &end, 10);
  if (errno == ERANGE || *end != '\0') {
    return false;
  }
  *out = (uint64_t)value;
  return true;
}

static bool parse_rowid(const char *text, char **end, int64_t *out) {
  const char *digits = text + (text[0] == '-');
  if (digits[0] < '0' || digits[0] > '9') {
    return false;
  }
  errno = 0;
  long long value = strtoll(text, end, 10);
  if (errno == ERANGE) {
    return false;
  }
  *out = (int64_t)value;
  return true;
}

static bool exec_ok(sqlite3 *db, const char *sql) {
  return sqlite3_exec(db, sql, NULL, NULL, NULL) == SQLITE_OK;
}

static bool prepare(sqlite3 *db, const char *sql, sqlite3_stmt **stmt) {
  return sqlite3_prepare_v2(db, sql, -1, stmt, NULL) == SQLITE_OK;
}

static int read_ranked(sqlite3_stmt *stmt, int64_t *ids, size_t cap,
                       size_t *n) {
  *n = 0U;
  int rc;
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    if (*n == cap) {
      return SQLITE_ERROR;
    }
    ids[(*n)++] = sqlite3_column_int64(stmt, 0);
  }
  return rc == SQLITE_DONE ? SQLITE_OK : rc;
}

static int cmd_version(void) {
  printf("{\"version\":\"%s\",\"upstream_commit\":\"%s\"}\n",
         SVE_HYBRID_VERSION, SVE_UPSTREAM_COMMIT);
  return 0;
}

static bool read_scalar(sqlite3 *db, const char *sql, double *value) {
  sqlite3_stmt *stmt = NULL;
  bool ok = prepare(db, sql, &stmt) && sqlite3_step(stmt) == SQLITE_ROW &&
            sqlite3_column_type(stmt, 0) != SQLITE_NULL;
  if (ok) {
    *value = sqlite3_column_double(stmt, 0);
    ok = sqlite3_step(stmt) == SQLITE_DONE;
  }
  sqlite3_finalize(stmt);
  return ok;
}

static bool test_bound_blob(sqlite3 *db, const char *ddl, int *step_rc) {
  sqlite3_stmt *stmt = NULL;
  const int8_t raw[] = {10, -20, 30, -40};
  bool ok = exec_ok(db, ddl) &&
            prepare(db, "INSERT INTO ti(rowid,v) VALUES(1,?1);", &stmt) &&
            sqlite3_bind_blob(stmt, 1, raw, (int)sizeof(raw), SQLITE_STATIC) == SQLITE_OK;
  if (ok) {
    *step_rc = sqlite3_step(stmt);
    ok = *step_rc == SQLITE_DONE || *step_rc == SQLITE_ERROR ||
         *step_rc == SQLITE_CONSTRAINT;
  }
  sqlite3_finalize(stmt);
  return ok;
}

static bool compare_ghost(sqlite3 *db, const char *ddl, double *count) {
  return exec_ok(db, ddl) && exec_ok(db, "BEGIN;") &&
         !exec_ok(db, "INSERT INTO t(rowid,e,c) VALUES(42,'[1,0,0,0]','bad');") &&
         exec_ok(db, "COMMIT;") &&
         read_scalar(db, "SELECT count(*) FROM t WHERE e MATCH '[1,0,0,0]' AND k=5;",
                     count);
}

static int cmd_repro_compare(void) {
  sqlite3 *up = NULL, *hy = NULL;
  sqlite3_stmt *wrap = NULL, *checked = NULL;
  double up_ghost = 0.0, hy_ghost = 0.0;
  int up_i8 = SQLITE_ERROR, hy_i8 = SQLITE_ERROR;
  double up_large = 0.0, up_tiny = 0.0, hy_large = 0.0, hy_tiny = 0.0;

  bool ok = sqlite3_open(":memory:", &up) == SQLITE_OK &&
            sqlite3_open(":memory:", &hy) == SQLITE_OK &&
            sqlite3_vec_init(up, NULL, NULL) == SQLITE_OK &&
            sve_hybrid_register(hy, NULL) == SQLITE_OK &&
            compare_ghost(up, "CREATE VIRTUAL TABLE t USING vec0(e float[4], +c integer);",
                          &up_ghost) &&
            compare_ghost(hy, "CREATE VIRTUAL TABLE t USING sve0(e float[4], +c integer);",
                          &hy_ghost) &&
            test_bound_blob(up, "CREATE VIRTUAL TABLE ti USING vec0(v int8[4]);", &up_i8) &&
            test_bound_blob(hy, "CREATE VIRTUAL TABLE ti USING sve0(v int8[4]);", &hy_i8) &&
            read_scalar(up, "SELECT vec_distance_l2('[1e20,0,0,0]','[0,0,0,0]');", &up_large) &&
            read_scalar(up, "SELECT vec_distance_l2('[1e-25,0,0,0]','[0,0,0,0]');", &up_tiny) &&
            read_scalar(hy, "SELECT sve_distance_l2('[1e20,0,0,0]','[0,0,0,0]');", &hy_large) &&
            read_scalar(hy, "SELECT sve_distance_l2('[1e-25,0,0,0]','[0,0,0,0]');", &hy_tiny) &&
            prepare(up, "SELECT vec_to_json(vec_add(vec_int8('[100,100]'),"
                        "vec_int8('[50,50]')));", &wrap) &&
            prepare(hy, "SELECT sve_to_json(sve_add(sve_int8('[100,100]'),"
                        "sve_int8('[50,50]')));", &checked);
  if (ok) {
    int up_rc = sqlite3_step(wrap);
    const char *text = up_rc == SQLITE_ROW ? (const char *)sqlite3_column_text(wrap, 0) : NULL;
    int hy_rc = sqlite3_step(checked);
    bool upstream_wrap = text && strcmp(text, "[-106,-106]") == 0;
    bool hybrid_rejected = hy_rc == SQLITE_ERROR &&
                           strstr(sqlite3_errmsg(hy), "overflow") != NULL;
    ok = (up_rc == SQLITE_ROW || up_rc == SQLITE_ERROR) &&
         (hy_rc == SQLITE_ROW || hy_rc == SQLITE_ERROR) &&
         isfinite(up_tiny) && isfinite(hy_large) && isfinite(hy_tiny);
    if (ok) {
      printf(
          "{\n"
          "  \"ghost_row_333\": {\"upstream_ghost_count\": %.0f, "
          "\"hybrid_ghost_count\": %.0f},\n"
          "  \"bound_int8_blob_361\": {\"upstream_rejected\": %s, "
          "\"hybrid_accepted\": %s},\n"
          "  \"l2_scale_334\": {\"upstream_large_is_inf\": %s, "
          "\"upstream_tiny\": %.3g, \"hybrid_large\": %.6g, \"hybrid_tiny\": %.6g},\n"
          "  \"vec_add_overflow_363\": {\"upstream_silent_wrap\": \"%s\", "
          "\"upstream_emitted\": %s, \"hybrid_rejected\": %s}\n"
          "}\n",
          up_ghost, hy_ghost, up_i8 != SQLITE_DONE ? "true" : "false",
          hy_i8 == SQLITE_DONE ? "true" : "false",
          isinf(up_large) ? "true" : "false", up_tiny, hy_large, hy_tiny,
          upstream_wrap ? "[-106,-106]" : "", text ? "true" : "false",
          hybrid_rejected ? "true" : "false");
    }
  }
  sqlite3_finalize(wrap);
  sqlite3_finalize(checked);
  sqlite3_close(up);
  sqlite3_close(hy);
  if (!ok) {
    fputs("Could not complete the upstream comparison.\n", stderr);
  }
  return ok ? 0 : 1;
}

static uint64_t xorshift64(uint64_t *s) {
  uint64_t x = *s;
  x ^= x << 13U;
  x ^= x >> 7U;
  x ^= x << 17U;
  *s = x;
  return x;
}

static float rand_unit_f32(uint64_t *s) {
  uint64_t r = xorshift64(s);
  return (float)(((double)(r & 0xFFFFFFU) / 16777216.0) * 2.0 - 1.0);
}

static void gen_norm_vec(uint64_t *rng, const float *centers, size_t n_centers,
                         size_t cluster_id, float *out, size_t dims) {
  const float *c = centers + (cluster_id % n_centers) * dims;
  double norm = 0.0;
  for (size_t d = 0U; d < dims; d++) {
    float v = c[d] + 0.25f * rand_unit_f32(rng);
    out[d] = v;
    norm += (double)v * (double)v;
  }
  double inv = norm > 0.0 ? 1.0 / sqrt(norm) : 1.0;
  for (size_t d = 0U; d < dims; d++) {
    out[d] = (float)((double)out[d] * inv);
  }
}

static double recall_at_k(const sve_search_hit_t *gt, size_t n_gt,
                          const int64_t *pred, size_t n_pred) {
  size_t hits = 0U;
  for (size_t i = 0U; i < n_gt; i++) {
    for (size_t j = 0U; j < n_pred; j++) {
      if (gt[i].rowid == pred[j]) {
        hits++;
        break;
      }
    }
  }
  return (double)hits / (double)n_gt;
}

typedef struct ann_fixture {
  size_t n;
  size_t dims;
  size_t k;
  size_t nq;
  const float *vecs;
  const float *qvecs;
  const int64_t *cats;
  sve_hnsw_t *exact;
  const int64_t *allow;
  size_t n_allow;
} ann_fixture_t;

typedef struct ann_result {
  double build_ms;
  latency_t lat;
  latency_t filtered_lat;
  double recall;
  double filtered_recall;
} ann_result_t;

static bool bench_ann_queries(sqlite3 *db, const ann_fixture_t *f,
                              bool compare, bool filtered, latency_t *lat,
                              double *recall) {
  const char *sql = filtered
      ? "SELECT rowid, distance FROM v WHERE emb MATCH ?1 AND k=?2 AND cat=0;"
      : "SELECT rowid, distance FROM v WHERE emb MATCH ?1 AND k=?2;";
  sqlite3_stmt *stmt = NULL;
  double *times = (double *)malloc(f->nq * sizeof(*times));
  int64_t *ids = (int64_t *)malloc(f->k * sizeof(*ids));
  sve_search_hit_t *truth = (sve_search_hit_t *)malloc(f->k * sizeof(*truth));
  bool ok = times && ids && truth && prepare(db, sql, &stmt);
  double total = 0.0;
  size_t available = filtered ? f->n_allow : f->n;
  size_t expected = f->k < available ? f->k : available;
  for (size_t q = 0U; ok && q < f->nq; q++) {
    const float *vec = f->qvecs + q * f->dims;
    size_t n_truth = 0U;
    if (compare && (sve_hnsw_exact_search(
           f->exact, vec, f->k, filtered ? f->allow : NULL, f->n_allow,
           truth, &n_truth) != SVE_OK || n_truth != expected)) {
      ok = false;
      break;
    }
    size_t n_ids = 0U;
    double start = now_ms();
    ok = sqlite3_reset(stmt) == SQLITE_OK &&
         sqlite3_bind_blob(stmt, 1, vec, (int)(f->dims * sizeof(float)),
                           SQLITE_STATIC) == SQLITE_OK &&
         sqlite3_bind_int64(stmt, 2, (sqlite3_int64)f->k) == SQLITE_OK &&
         read_ranked(stmt, ids, f->k, &n_ids) == SQLITE_OK &&
         n_ids == expected;
    times[q] = now_ms() - start;
    if (ok && compare) {
      total += recall_at_k(truth, n_truth, ids, n_ids);
    }
  }
  if (ok) {
    *lat = latency_summary(times, f->nq);
    *recall = total / (double)f->nq;
  }
  sqlite3_finalize(stmt);
  free(times);
  free(ids);
  free(truth);
  return ok;
}

static bool bench_ann_mode(const ann_fixture_t *f, const char *index_kind,
                            ann_result_t *out) {
  sqlite3 *db = NULL;
  sqlite3_stmt *ins = NULL;
  bool upstream = index_kind == NULL;
  bool ok = sqlite3_open(":memory:", &db) == SQLITE_OK;
  if (ok) {
    ok = (upstream ? sqlite3_vec_init(db, NULL, NULL)
                   : sve_hybrid_register(db, NULL)) == SQLITE_OK;
  }
  char ddl[256];
  if (upstream) {
    snprintf(ddl, sizeof(ddl), "CREATE VIRTUAL TABLE v USING vec0("
             "emb float[%zu] distance_metric=cosine, cat integer);", f->dims);
  } else {
    snprintf(ddl, sizeof(ddl), "CREATE VIRTUAL TABLE v USING sve0("
             "emb float[%zu] index=%s metric=cosine m=16 ef_construction=100"
             " ef_search=64, cat integer);", f->dims, index_kind);
  }
  ok = ok && exec_ok(db, ddl) && exec_ok(db, "BEGIN;") &&
       prepare(db, "INSERT INTO v(rowid,emb,cat) VALUES(?1,?2,?3);", &ins);
  double start = now_ms();
  for (size_t i = 0U; ok && i < f->n; i++) {
    ok = sqlite3_reset(ins) == SQLITE_OK &&
         sqlite3_bind_int64(ins, 1, (sqlite3_int64)(i + 1U)) == SQLITE_OK &&
         sqlite3_bind_blob(ins, 2, f->vecs + i * f->dims,
                           (int)(f->dims * sizeof(float)), SQLITE_STATIC) == SQLITE_OK &&
         sqlite3_bind_int64(ins, 3, f->cats[i]) == SQLITE_OK &&
         sqlite3_step(ins) == SQLITE_DONE;
  }
  sqlite3_finalize(ins);
  ok = ok && exec_ok(db, "COMMIT;");
  out->build_ms = now_ms() - start;
  ok = ok && bench_ann_queries(db, f, !upstream, false, &out->lat, &out->recall);
  if (ok && (upstream || strcmp(index_kind, "hnsw") == 0)) {
    ok = bench_ann_queries(db, f, !upstream, true, &out->filtered_lat,
                           &out->filtered_recall);
  }
  sqlite3_close(db);
  return ok;
}

static int cmd_ann_bench(int argc, char **argv) {
  ann_fixture_t f = {.n = 10000U, .dims = 128U, .k = 10U, .nq = 100U};
  size_t filter_pct = 5U;
  uint64_t rng = 42ULL;
  for (int i = 2; i < argc; i += 2) {
    uint64_t value;
    if (i + 1 >= argc || !parse_u64(argv[i + 1], &value)) {
      return 1;
    }
    if (strcmp(argv[i], "--seed") == 0) {
      rng = value ? value : 42ULL;
    } else {
      if (value == 0U || value > SIZE_MAX) {
        return 1;
      }
      size_t parsed = (size_t)value;
      if (strcmp(argv[i], "--n") == 0) {
        f.n = parsed;
      } else if (strcmp(argv[i], "--dims") == 0) {
        f.dims = parsed;
      } else if (strcmp(argv[i], "--queries") == 0) {
        f.nq = parsed;
      } else if (strcmp(argv[i], "--k") == 0) {
        f.k = parsed;
      } else if (strcmp(argv[i], "--filter-pct") == 0) {
        filter_pct = parsed;
      } else {
        return 1;
      }
    }
  }
  if (f.dims > SVE_MAX_TABLE_DIMS || f.k > SVE_MAX_K || filter_pct > 100U ||
      f.n > INT64_MAX || f.n > SIZE_MAX / sizeof(int64_t) ||
      f.nq > SIZE_MAX / sizeof(double) ||
      f.n > SIZE_MAX / sizeof(float) / f.dims ||
      f.nq > SIZE_MAX / sizeof(float) / f.dims) {
    return 1;
  }

  const size_t n_centers = 32U;
  float *centers = (float *)malloc(n_centers * f.dims * sizeof(float));
  float *vecs = (float *)malloc(f.n * f.dims * sizeof(float));
  float *qvecs = (float *)malloc(f.nq * f.dims * sizeof(float));
  int64_t *cats = (int64_t *)malloc(f.n * sizeof(int64_t));
  int64_t *allowed = (int64_t *)malloc(f.n * sizeof(int64_t));
  f.vecs = vecs;
  f.qvecs = qvecs;
  f.cats = cats;
  f.allow = allowed;
  sve_hnsw_config_t cfg = sve_hnsw_default_config(
      f.dims, SVE_ELEM_FLOAT32, SVE_METRIC_COSINE, SVE_INDEX_FLAT);
  f.exact = sve_hnsw_create(&cfg);
  bool ok = centers && vecs && qvecs && cats && allowed && f.exact;
  if (ok) {
    for (size_t i = 0U; i < n_centers * f.dims; i++) {
      centers[i] = rand_unit_f32(&rng);
    }
    size_t n_buckets = 100U / filter_pct;
    for (size_t i = 0U; ok && i < f.n; i++) {
      float *vec = vecs + i * f.dims;
      gen_norm_vec(&rng, centers, n_centers, i % n_centers, vec, f.dims);
      int64_t id = (int64_t)(i + 1U);
      cats[i] = (int64_t)(i % n_buckets);
      if (cats[i] == 0) {
        allowed[f.n_allow++] = id;
      }
      ok = sve_hnsw_insert(f.exact, id, vec) == SVE_OK;
    }
    for (size_t q = 0U; ok && q < f.nq; q++) {
      gen_norm_vec(&rng, centers, n_centers, q % n_centers, qvecs + q * f.dims, f.dims);
    }
  }

  ann_result_t up = {0}, hy = {0}, sq = {0};
  ok = ok && bench_ann_mode(&f, NULL, &up) &&
       bench_ann_mode(&f, "hnsw", &hy) && bench_ann_mode(&f, "hnsw_sq8", &sq);
  if (ok) {
    printf(
        "{\n"
        "  \"n\": %zu, \"dims\": %zu, \"k\": %zu, \"queries\": %zu, "
        "\"filter_pct\": %zu, \"filter_docs\": %zu,\n"
        "  \"upstream_vec0\": {\"build_ms\": %.2f, \"p50_ms\": %.4f, "
        "\"p99_ms\": %.4f, \"filtered_p99_ms\": %.4f},\n"
        "  \"sve0_hnsw\": {\"build_ms\": %.2f, \"p50_ms\": %.4f, "
        "\"p99_ms\": %.4f, \"filtered_p99_ms\": %.4f, \"recall_at_k\": %.4f, "
        "\"filtered_recall_at_k\": %.4f, \"speedup_p50_x\": %.2f, "
        "\"speedup_p99_x\": %.2f},\n"
        "  \"sve0_hnsw_sq8\": {\"build_ms\": %.2f, \"p50_ms\": %.4f, "
        "\"p99_ms\": %.4f, \"recall_at_k\": %.4f, \"speedup_p50_x\": %.2f}\n"
        "}\n",
        f.n, f.dims, f.k, f.nq, filter_pct, f.n_allow,
        up.build_ms, up.lat.p50, up.lat.p99, up.filtered_lat.p99,
        hy.build_ms, hy.lat.p50, hy.lat.p99, hy.filtered_lat.p99, hy.recall,
        hy.filtered_recall, hy.lat.p50 > 0.0 ? up.lat.p50 / hy.lat.p50 : 0.0,
        hy.lat.p99 > 0.0 ? up.lat.p99 / hy.lat.p99 : 0.0,
        sq.build_ms, sq.lat.p50, sq.lat.p99, sq.recall,
        sq.lat.p50 > 0.0 ? up.lat.p50 / sq.lat.p50 : 0.0);
  } else {
    fputs("ANN benchmark failed during index build or search.\n", stderr);
  }
  sve_hnsw_destroy(f.exact);
  free(allowed);
  free(cats);
  free(qvecs);
  free(vecs);
  free(centers);
  return ok ? 0 : 1;
}

static int cmd_privacy_audit(void) {
  static const char *const k_secrets[15] = {
      "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload99",
      "imsi=460001234567890 registered on cell 7",
      "imei: 867530912345678 baseband ready",
      "msisdn=+86" "13800" "138000 alert",
      "contact mobile: 13912" "345678 asap",
      "api_key=\"live_token_9876543210abc\"",
      "sk-" "proj-abcDEF1234567890xyz",
      "github token ghp" "_AbCdEfGhIjKlMnOpQrSt1234",
      "aws key AKIA" "IOSFODNN7EXAMPLE",
      "dsn postgresql+asyncpg://svc:SuperSecret99@db.internal:5432/vec",
      "quic_ticket=qt_a1b2c3d4e5f6g7h8i9j0",
      "password = \"trickyPass!8899\"",
      "-----" "BEGIN RSA " "PRIVATE KEY-----\nMIIEowIBAAKCAQEAtest\n-----" "END RSA " "PRIVATE KEY-----",
      "backup phone: +86-138-1234-5678",
      "auth_token: tok_edge_991122334455"};

  static const char *const k_benign[15] = {
      "UUID 550e8400-e29b-41d4-a716-446655440000 verified cleanly",
      "SHA256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "ISO date 2026-11-25T12:00:00Z logged",
      "order_id=13800" "138000 shipped via rail freight",
      "Bearer authentication specification RFC 6750 defines scopes",
      "latency p99 = 0.42 ms across 10000 vectors",
      "SQLite WAL checkpoint completed on sector 12",
      "embedding dimension D=384 cosine metric normalized",
      "tx_hash=13912" "345678 confirmed at block height 90012",
      "HNSW ef_construction=100 M=16 recall@10 = 0.992",
      "temperature sensor calibrated at 23.45 C",
      "POSIX thread pool worker-03 idle",
      "chunk compression ratio 3.04x on telemetry stream",
      "batch 13700" "137000 passed checksum verification",
      "HTTP 200 OK healthcheck elapsed 1 ms"};

  char buf[4096];
  size_t secret_hits = 0U;
  size_t benign_fps = 0U;
  static const char *const k_removed_suffix[15] = {
      "payload99", "567890", "345678", "13800" "138000", "13912" "345678",
      "76543210abc", "7890xyz", "MnOpQrSt1234", "7EXAMPLE",
      "Secret99", "h8i9j0", "8899", "AQEAtest", "1234-5678", "22334455"};

  for (size_t i = 0U; i < 15U; i++) {
    size_t nred = 0U;
    if (sve_privacy_scrub(k_secrets[i], strlen(k_secrets[i]), buf, sizeof(buf),
                          &nred) == SVE_OK &&
        nred > 0U && strstr(buf, "[REDACTED]") != NULL &&
        strstr(buf, k_removed_suffix[i]) == NULL) {
      secret_hits++;
    }
  }
  for (size_t i = 0U; i < 15U; i++) {
    size_t nred = 0U;
    if (sve_privacy_scrub(k_benign[i], strlen(k_benign[i]), buf, sizeof(buf),
                          &nred) != SVE_OK ||
        nred > 0U || strcmp(buf, k_benign[i]) != 0) {
      benign_fps++;
    }
  }

  printf("{\"secret_recall\":\"%zu/15\",\"benign_fp\":\"%zu/15\","
         "\"passed\":%s}\n",
         secret_hits, benign_fps,
         (secret_hits == 15U && benign_fps == 0U) ? "true" : "false");
  return (secret_hits == 15U && benign_fps == 0U) ? 0 : 1;
}

/*
 * hybrid-eval: reads a TSV package emitted by the Python benchmark runner:
 * Line format per doc/query in simple TSV:
 *   DOC <rowid>\t<hex_f32_le>\t<clean_text>
 *   QRY <qid>\t<hex_f32>\t<query_text>\t<comma_sep_pos_rowids>
 */
static int hex_digit(char c) {
  if (c >= '0' && c <= '9') {
    return c - '0';
  }
  if (c >= 'a' && c <= 'f') {
    return c - 'a' + 10;
  }
  if (c >= 'A' && c <= 'F') {
    return c - 'A' + 10;
  }
  return -1;
}

static bool hex_decode_f32(const char *hex, size_t dims, float *out) {
  if (strlen(hex) != dims * 8U) {
    return false;
  }
  for (size_t i = 0U; i < dims; i++) {
    uint32_t bits = 0U;
    for (size_t j = 0U; j < 4U; j++) {
      int hi = hex_digit(hex[i * 8U + j * 2U]);
      int lo = hex_digit(hex[i * 8U + j * 2U + 1U]);
      if (hi < 0 || lo < 0) {
        return false;
      }
      bits |= (uint32_t)(hi * 16 + lo) << (j * 8U);
    }
    memcpy(&out[i], &bits, sizeof(bits));
  }
  return sve_f32_array_is_finite(out, dims);
}

typedef struct eval_query {
  float *vec;
  char *text;
  int64_t *pos_ids;
  size_t n_pos;
} eval_query_t;

static double compute_ndcg10(const int64_t *ranked, size_t n_ranked,
                             const int64_t *pos, size_t n_pos) {
  if (n_pos == 0U) {
    return 0.0;
  }
  double dcg = 0.0;
  size_t lim = n_ranked < 10U ? n_ranked : 10U;
  for (size_t i = 0U; i < lim; i++) {
    for (size_t j = 0U; j < n_pos; j++) {
      if (ranked[i] == pos[j]) {
        dcg += 1.0 / log2((double)(i + 2U));
        break;
      }
    }
  }
  double idcg = 0.0;
  size_t ideal_k = n_pos < 10U ? n_pos : 10U;
  for (size_t i = 0U; i < ideal_k; i++) {
    idcg += 1.0 / log2((double)(i + 2U));
  }
  return idcg > 0.0 ? dcg / idcg : 0.0;
}

static double compute_mrr10(const int64_t *ranked, size_t n_ranked,
                            const int64_t *pos, size_t n_pos) {
  size_t lim = n_ranked < 10U ? n_ranked : 10U;
  for (size_t i = 0U; i < lim; i++) {
    for (size_t j = 0U; j < n_pos; j++) {
      if (ranked[i] == pos[j]) {
        return 1.0 / (double)(i + 1U);
      }
    }
  }
  return 0.0;
}

static double compute_recall10(const int64_t *ranked, size_t n_ranked,
                               const int64_t *pos, size_t n_pos) {
  if (n_pos == 0U) {
    return 0.0;
  }
  size_t lim = n_ranked < 10U ? n_ranked : 10U;
  size_t hits = 0U;
  for (size_t j = 0U; j < n_pos; j++) {
    for (size_t i = 0U; i < lim; i++) {
      if (ranked[i] == pos[j]) {
        hits++;
        break;
      }
    }
  }
  return (double)hits / (double)n_pos;
}

static int read_tsv_line(FILE *file, char *buf, size_t capacity) {
  size_t n = 0U;
  int c = 0;
  while ((c = fgetc(file)) != EOF && c != '\n') {
    if (c == 0 || n == capacity - 1U) {
      return -1;
    }
    buf[n++] = (char)c;
  }
  if (c == EOF && (ferror(file) || n == 0U)) {
    return ferror(file) ? -1 : 0;
  }
  if (n > 0U && buf[n - 1U] == '\r') {
    n--;
  }
  buf[n] = '\0';
  return 1;
}

static bool split_tsv(char *line, char **fields, size_t count) {
  for (size_t i = 1U; i < count; i++) {
    char *tab = strchr(line, '\t');
    if (!tab) {
      return false;
    }
    *tab = '\0';
    fields[i - 1U] = line;
    line = tab + 1U;
  }
  fields[count - 1U] = line;
  return !strchr(line, '\t');
}

static int cmp_i64(const void *a, const void *b) {
  int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
  return (x > y) - (x < y);
}

static bool load_eval_query(eval_query_t *query, const float *vec, int vec_bytes,
                            const char *text, const char *positives) {
  size_t text_len = strlen(text);
  if (text_len > SVE_MAX_TEXT_BYTES || *positives == '\0') {
    return false;
  }
  size_t count = 1U;
  for (const char *c = positives; *c; c++) {
    if (*c == ',') {
      count++;
    }
  }
  query->vec = (float *)malloc((size_t)vec_bytes);
  query->text = (char *)malloc(text_len + 1U);
  query->pos_ids = (int64_t *)malloc(count * sizeof(*query->pos_ids));
  if (!query->vec || !query->text || !query->pos_ids) {
    return false;
  }
  memcpy(query->vec, vec, (size_t)vec_bytes);
  memcpy(query->text, text, text_len + 1U);
  for (size_t i = 0U; i < count; i++) {
    char *end = NULL;
    if (!parse_rowid(positives, &end, &query->pos_ids[i]) ||
        *end != (i + 1U < count ? ',' : '\0')) {
      return false;
    }
    positives = end + 1;
  }
  qsort(query->pos_ids, count, sizeof(*query->pos_ids), cmp_i64);
  for (size_t i = 1U; i < count; i++) {
    if (query->pos_ids[i - 1U] == query->pos_ids[i]) {
      return false;
    }
  }
  query->n_pos = count;
  return true;
}

static bool insert_eval_doc(sqlite3_stmt *stmt, int64_t id, const float *vec,
                             int vec_bytes, const char *text) {
  if (sqlite3_reset(stmt) != SQLITE_OK ||
      sqlite3_bind_int64(stmt, 1, id) != SQLITE_OK) {
    return false;
  }
  int col = 2;
  if (vec && sqlite3_bind_blob(stmt, col++, vec, vec_bytes, SQLITE_TRANSIENT) != SQLITE_OK) {
    return false;
  }
  if (text && sqlite3_bind_text(stmt, col, text, -1, SQLITE_TRANSIENT) != SQLITE_OK) {
    return false;
  }
  return sqlite3_step(stmt) == SQLITE_DONE;
}

typedef struct eval_mode {
  const char *name;
  const char *sql;
  sqlite3_stmt *stmt;
  double *times;
  double ndcg;
  double mrr;
  double recall;
} eval_mode_t;

static int cmd_hybrid_eval(int argc, char **argv) {
  const char *path = NULL;
  size_t dims = 0U;
  for (int i = 2; i < argc; i += 2) {
    uint64_t parsed;
    if (i + 1 >= argc) {
      return 1;
    }
    if (strcmp(argv[i], "--input") == 0) {
      path = argv[i + 1];
    } else if (strcmp(argv[i], "--dims") == 0 &&
               parse_u64(argv[i + 1], &parsed) && parsed <= SVE_MAX_TABLE_DIMS) {
      dims = (size_t)parsed;
    } else {
      return 1;
    }
  }
  if (!path || dims == 0U) {
    return 1;
  }
  FILE *fp = NULL;
#ifdef _MSC_VER
  if (fopen_s(&fp, path, "rb") != 0) {
    fp = NULL;
  }
#else
  fp = fopen(path, "rb");
#endif
  if (!fp) {
    fputs("Could not open benchmark input.\n", stderr);
    return 1;
  }

  sqlite3 *up = NULL, *hy = NULL;
  sqlite3_stmt *up_ins = NULL, *hy_ins = NULL, *bm_ins = NULL, *relevant = NULL;
  eval_mode_t modes[] = {
      {.name = "upstream_vec0",
       .sql = "SELECT rowid FROM uv WHERE emb MATCH ?1 AND k=10;"},
      {.name = "fts5_bm25",
       .sql = "SELECT rowid FROM bm_only WHERE bm_only MATCH ?1 "
              "ORDER BY bm25(bm_only), rowid LIMIT 10;"},
      {.name = "sve0_hnsw",
       .sql = "SELECT rowid FROM hv WHERE emb MATCH ?1 AND k=10;"},
      {.name = "sve0_hybrid_rrf",
       .sql = "SELECT rowid FROM hv WHERE emb MATCH ?1 AND text_query MATCH ?2 AND k=10;"},
      {.name = "sve0_hybrid_convex",
       .sql = "SELECT rowid FROM hv WHERE emb MATCH ?1 AND text_query MATCH ?2 "
              "AND hybrid_alpha=0.25 AND k=10;"},
  };
  const size_t n_modes = sizeof(modes) / sizeof(modes[0]);
  const size_t line_cap = 1024U * 1024U;
  char *line = (char *)malloc(line_cap);
  int vec_bytes = (int)(dims * sizeof(float));
  float *vec = (float *)malloc((size_t)vec_bytes);
  eval_query_t *queries = NULL;
  size_t n_queries = 0U, capacity = 0U, n_docs = 0U, line_count = 0U;
  size_t line_failure = 0U;

  bool ok = line && vec && sqlite3_open(":memory:", &up) == SQLITE_OK &&
            sqlite3_open(":memory:", &hy) == SQLITE_OK &&
            sqlite3_vec_init(up, NULL, NULL) == SQLITE_OK &&
            sve_hybrid_register(hy, NULL) == SQLITE_OK;
  char ddl_up[256], ddl_hy[256];
  snprintf(ddl_up, sizeof(ddl_up),
           "CREATE VIRTUAL TABLE uv USING vec0(emb float[%zu] distance_metric=cosine);",
           dims);
  snprintf(ddl_hy, sizeof(ddl_hy),
           "CREATE VIRTUAL TABLE hv USING sve0(emb float[%zu] index=hnsw "
           "metric=cosine m=16 ef_construction=100 ef_search=64, body text fulltext);",
           dims);
  ok = ok && exec_ok(up, ddl_up) && exec_ok(hy, ddl_hy) &&
       exec_ok(hy, "CREATE VIRTUAL TABLE bm_only USING fts5(doc_text, tokenize='unicode61');") &&
       exec_ok(up, "BEGIN;") && exec_ok(hy, "BEGIN;") &&
       prepare(up, "INSERT INTO uv(rowid,emb) VALUES(?1,?2);", &up_ins) &&
       prepare(hy, "INSERT INTO hv(rowid,emb,body) VALUES(?1,?2,?3);", &hy_ins) &&
       prepare(hy, "INSERT INTO bm_only(rowid,doc_text) VALUES(?1,?2);", &bm_ins);

  while (ok) {
    int result = read_tsv_line(fp, line, line_cap);
    if (result == 0) {
      break;
    }
    line_count++;
    bool doc = result > 0 && strncmp(line, "DOC\t", 4U) == 0;
    bool query = result > 0 && strncmp(line, "QRY\t", 4U) == 0;
    char *fields[4] = {0};
    int64_t id = 0;
    char *end = NULL;
    if ((!doc && !query) || !split_tsv(line + 4U, fields, doc ? 3U : 4U) ||
        !parse_rowid(fields[0], &end, &id) || *end != '\0' ||
        !hex_decode_f32(fields[1], dims, vec)) {
      ok = false;
    } else if (doc) {
      ok = insert_eval_doc(up_ins, id, vec, vec_bytes, NULL) &&
           insert_eval_doc(hy_ins, id, vec, vec_bytes, fields[2]) &&
           insert_eval_doc(bm_ins, id, NULL, vec_bytes, fields[2]);
      if (ok) {
        n_docs++;
      }
    } else {
      if (n_queries == capacity) {
        size_t next = capacity == 0U ? 64U : capacity * 2U;
        eval_query_t *tmp = NULL;
        if (capacity <= SIZE_MAX / (2U * sizeof(*queries))) {
          tmp = (eval_query_t *)realloc(queries, next * sizeof(*queries));
        }
        if (!tmp) {
          ok = false;
          break;
        }
        capacity = next;
        queries = tmp;
      }
      eval_query_t *queued = &queries[n_queries++];
      memset(queued, 0, sizeof(*queued));
      ok = load_eval_query(queued, vec, vec_bytes, fields[2], fields[3]);
    }
    if (!ok) {
      line_failure = line_count;
    }
  }
  fclose(fp);
  sqlite3_finalize(up_ins);
  sqlite3_finalize(hy_ins);
  sqlite3_finalize(bm_ins);
  ok = ok && n_docs > 0U && n_queries > 0U && exec_ok(up, "COMMIT;") &&
       exec_ok(hy, "COMMIT;") &&
       prepare(hy, "SELECT id FROM hv_rows WHERE id=?1;", &relevant);
  for (size_t m = 0U; ok && m < n_modes; m++) {
    modes[m].times = (double *)malloc(n_queries * sizeof(*modes[m].times));
    ok = modes[m].times && prepare(m == 0U ? up : hy, modes[m].sql, &modes[m].stmt);
  }

  size_t expected = n_docs < 10U ? n_docs : 10U;
  for (size_t q = 0U; ok && q < n_queries; q++) {
    const eval_query_t *query = &queries[q];
    for (size_t j = 0U; ok && j < query->n_pos; j++) {
      ok = sqlite3_reset(relevant) == SQLITE_OK &&
           sqlite3_bind_int64(relevant, 1, query->pos_ids[j]) == SQLITE_OK &&
           sqlite3_step(relevant) == SQLITE_ROW &&
           sqlite3_step(relevant) == SQLITE_DONE;
    }
    char sanitized[4096];
    ok = ok && sve_fts5_sanitize_query(query->text, 32U, sanitized, sizeof(sanitized)) == SVE_OK;
    for (size_t m = 0U; ok && m < n_modes; m++) {
      eval_mode_t *mode = &modes[m];
      sqlite3_stmt *stmt = mode->stmt;
      int64_t ids[10];
      size_t n = 0U;
      double start = now_ms();
      if (m != 1U || sanitized[0] != '\0') {
        ok = sqlite3_reset(stmt) == SQLITE_OK;
        if (ok && m == 1U) {
          ok = sqlite3_bind_text(stmt, 1, sanitized, -1, SQLITE_TRANSIENT) == SQLITE_OK;
        } else if (ok) {
          ok = sqlite3_bind_blob(stmt, 1, query->vec, vec_bytes, SQLITE_STATIC) == SQLITE_OK;
          if (ok && m >= 3U) {
            ok = sqlite3_bind_text(stmt, 2, query->text, -1, SQLITE_STATIC) == SQLITE_OK;
          }
        }
        ok = ok && read_ranked(stmt, ids, 10U, &n) == SQLITE_OK &&
             (m == 1U || n == expected);
      }
      mode->times[q] = now_ms() - start;
      if (ok) {
        mode->ndcg += compute_ndcg10(ids, n, query->pos_ids, query->n_pos);
        mode->mrr += compute_mrr10(ids, n, query->pos_ids, query->n_pos);
        mode->recall += compute_recall10(ids, n, query->pos_ids, query->n_pos);
      }
    }
  }

  if (ok) {
    printf("{\n  \"docs\": %zu, \"queries\": %zu, \"dims\": %zu,\n",
           n_docs, n_queries, dims);
    for (size_t m = 0U; m < n_modes; m++) {
      eval_mode_t *mode = &modes[m];
      latency_t lat = latency_summary(mode->times, n_queries);
      double denom = (double)n_queries;
      printf("  \"%s\": {\"ndcg10\": %.4f, \"mrr10\": %.4f, "
             "\"recall10\": %.4f, \"p50_ms\": %.4f, \"p99_ms\": %.4f}%s\n",
             mode->name, mode->ndcg / denom, mode->mrr / denom, mode->recall / denom,
             lat.p50, lat.p99, m + 1U == n_modes ? "" : ",");
    }
    puts("}");
  } else if (line_failure != 0U) {
    fprintf(stderr, "Invalid or unreadable benchmark record at line %zu.\n", line_failure);
  } else {
    fputs("Hybrid benchmark failed during setup, ingestion, or evaluation.\n", stderr);
  }
  for (size_t m = 0U; m < n_modes; m++) {
    sqlite3_finalize(modes[m].stmt);
    free(modes[m].times);
  }
  sqlite3_finalize(relevant);
  for (size_t q = 0U; q < n_queries; q++) {
    free(queries[q].vec);
    free(queries[q].text);
    free(queries[q].pos_ids);
  }
  free(queries);
  free(line);
  free(vec);
  sqlite3_close(up);
  sqlite3_close(hy);
  return ok ? 0 : 1;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr,
            "Usage: sve_hybrid_cli "
            "<version|repro-compare|ann-bench|hybrid-eval|privacy-audit>\n");
    return 1;
  }
  if (strcmp(argv[1], "version") == 0) {
    return cmd_version();
  }
  if (strcmp(argv[1], "repro-compare") == 0) {
    return cmd_repro_compare();
  }
  if (strcmp(argv[1], "ann-bench") == 0) {
    return cmd_ann_bench(argc, argv);
  }
  if (strcmp(argv[1], "hybrid-eval") == 0) {
    return cmd_hybrid_eval(argc, argv);
  }
  if (strcmp(argv[1], "privacy-audit") == 0) {
    return cmd_privacy_audit();
  }
  fprintf(stderr, "Unknown subcommand: %s\n", argv[1]);
  return 1;
}
