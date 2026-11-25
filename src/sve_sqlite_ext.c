#include "sve_hybrid/sve_sqlite_ext.h"
#include "sve_hybrid/sve_config.h"
#include "sve_hybrid/sve_numeric.h"
#include "sve_hybrid/sve_vtab_guard.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SQLITE_CORE
SQLITE_EXTENSION_INIT1
#endif

#ifndef SQLITE_SUBTYPE
#define SQLITE_SUBTYPE 0x000100000
#endif

#ifndef SQLITE_RESULT_SUBTYPE
#define SQLITE_RESULT_SUBTYPE 0x001000000
#endif

#define SVE_JSON_SUBTYPE 74

typedef struct scalar_vec {
  void *data;
  size_t dims;
  sve_elem_type_t elem_type;
  bool owned;
} scalar_vec_t;

static void free_scalar_vec(scalar_vec_t *v) {
  if (v && v->owned && v->data) {
    free(v->data);
  }
  if (v) {
    v->data = NULL;
    v->dims = 0U;
    v->owned = false;
  }
}

static int extract_scalar_vec(sqlite3_value *val, scalar_vec_t *out,
                              char **pzErr) {
  memset(out, 0, sizeof(*out));
  int vt = sqlite3_value_type(val);
  int st = sqlite3_value_subtype(val);

  if (vt == SQLITE_BLOB) {
    int nbytes = sqlite3_value_bytes(val);
    const void *blob = sqlite3_value_blob(val);
    if (!blob || nbytes <= 0) {
      *pzErr = sqlite3_mprintf("vector BLOB cannot be empty");
      return SQLITE_ERROR;
    }
    if (st == SVE_ELEM_INT8) {
      if ((size_t)nbytes > SVE_MAX_SCALAR_DIMS) {
        *pzErr = sqlite3_mprintf("int8 vector exceeds max dimensions");
        return SQLITE_ERROR;
      }
      void *copy = malloc((size_t)nbytes);
      if (!copy) {
        return SQLITE_NOMEM;
      }
      memcpy(copy, blob, (size_t)nbytes);
      out->data = copy;
      out->dims = (size_t)nbytes;
      out->elem_type = SVE_ELEM_INT8;
      out->owned = true;
      return SQLITE_OK;
    }
    if (st == SVE_ELEM_BIT) {
      if ((size_t)nbytes > SVE_MAX_SCALAR_DIMS / 8U) {
        *pzErr = sqlite3_mprintf("bit vector exceeds max dimensions");
        return SQLITE_ERROR;
      }
      void *copy = malloc((size_t)nbytes);
      if (!copy) {
        return SQLITE_NOMEM;
      }
      memcpy(copy, blob, (size_t)nbytes);
      out->data = copy;
      out->dims = (size_t)nbytes * 8U;
      out->elem_type = SVE_ELEM_BIT;
      out->owned = true;
      return SQLITE_OK;
    }
    if (st != 0 && st != SVE_ELEM_FLOAT32) {
      *pzErr = sqlite3_mprintf("unknown vector subtype: %d", st);
      return SQLITE_ERROR;
    }
    if ((nbytes % (int)sizeof(float)) != 0) {
      *pzErr = sqlite3_mprintf("float32 BLOB byte size must be divisible by 4");
      return SQLITE_ERROR;
    }
    size_t dims = (size_t)nbytes / sizeof(float);
    if (dims > SVE_MAX_SCALAR_DIMS) {
      *pzErr = sqlite3_mprintf("float32 vector exceeds max dimensions");
      return SQLITE_ERROR;
    }
    float *copy = (float *)malloc((size_t)nbytes);
    if (!copy) {
      return SQLITE_NOMEM;
    }
    memcpy(copy, blob, (size_t)nbytes);
    if (!sve_f32_array_is_finite(copy, dims)) {
      free(copy);
      *pzErr = sqlite3_mprintf("float32 BLOB contains non-finite values");
      return SQLITE_ERROR;
    }
    out->data = copy;
    out->dims = dims;
    out->elem_type = SVE_ELEM_FLOAT32;
    out->owned = true;
    return SQLITE_OK;
  }

  if (vt == SQLITE_TEXT && (st == 0 || st == SVE_JSON_SUBTYPE)) {
    const char *txt = (const char *)sqlite3_value_text(val);
    int tlen = sqlite3_value_bytes(val);
    if (!txt || tlen <= 0 || (size_t)tlen > SVE_MAX_JSON_BYTES) {
      *pzErr = sqlite3_mprintf("invalid or oversized JSON vector string");
      return SQLITE_ERROR;
    }
    float *fv = NULL;
    size_t dims = 0U;
    int rc = sve_parse_f32_json(txt, (size_t)tlen, SVE_MAX_TABLE_DIMS, &fv, &dims);
    if (rc == SVE_ERR_NOMEM) {
      return SQLITE_NOMEM;
    }
    if (rc != SVE_OK) {
      *pzErr = sqlite3_mprintf("invalid JSON float32 vector");
      return SQLITE_ERROR;
    }
    out->data = fv;
    out->dims = dims;
    out->elem_type = SVE_ELEM_FLOAT32;
    out->owned = true;
    return SQLITE_OK;
  }

  *pzErr = sqlite3_mprintf("expected vector BLOB or JSON TEXT");
  return SQLITE_ERROR;
}

static int extract_matching_pair(sqlite3_value *va, sqlite3_value *vb,
                                 scalar_vec_t *a, scalar_vec_t *b,
                                 char **pzErr) {
  int rc = extract_scalar_vec(va, a, pzErr);
  if (rc != SQLITE_OK) {
    return rc;
  }
  rc = extract_scalar_vec(vb, b, pzErr);
  if (rc != SQLITE_OK) {
    free_scalar_vec(a);
    return rc;
  }
  if (a->elem_type != b->elem_type) {
    free_scalar_vec(a);
    free_scalar_vec(b);
    *pzErr = sqlite3_mprintf("vector element type mismatch");
    return SQLITE_ERROR;
  }
  if (a->dims != b->dims) {
    *pzErr = sqlite3_mprintf("vector dimension mismatch: %llu vs %llu",
                             (sqlite3_uint64)a->dims, (sqlite3_uint64)b->dims);
    free_scalar_vec(a);
    free_scalar_vec(b);
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}

static void scalar_error(sqlite3_context *ctx, int rc, char *err,
                         const char *fallback) {
  if (rc == SQLITE_NOMEM) {
    sqlite3_result_error_nomem(ctx);
  } else {
    sqlite3_result_error(ctx, err ? err : fallback, -1);
  }
  sqlite3_free(err);
}

static void fn_sve_version(sqlite3_context *ctx, int argc,
                           sqlite3_value **argv) {
  (void)argc;
  (void)argv;
  sqlite3_result_text(ctx, SVE_HYBRID_VERSION, -1, SQLITE_STATIC);
}

static void fn_sve_f32(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  (void)argc;
  char *err = NULL;
  scalar_vec_t v;
  int rc = extract_scalar_vec(argv[0], &v, &err);
  if (rc != SQLITE_OK) {
    scalar_error(ctx, rc, err, "invalid f32 vector");
    return;
  }
  if (v.elem_type != SVE_ELEM_FLOAT32) {
    free_scalar_vec(&v);
    sqlite3_result_error(ctx, "sve_f32 requires float32 input", -1);
    return;
  }
  size_t nbytes = v.dims * sizeof(float);
  sqlite3_result_blob64(ctx, v.data, (sqlite3_uint64)nbytes, free);
  sqlite3_result_subtype(ctx, SVE_ELEM_FLOAT32);
}

static void fn_sve_int8(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  (void)argc;
  int vt = sqlite3_value_type(argv[0]);
  if (vt == SQLITE_BLOB) {
    int nbytes = sqlite3_value_bytes(argv[0]);
    const void *blob = sqlite3_value_blob(argv[0]);
    if (!blob || nbytes <= 0 || (size_t)nbytes > SVE_MAX_SCALAR_DIMS) {
      sqlite3_result_error(ctx, "invalid int8 vector BLOB", -1);
      return;
    }
    void *copy = malloc((size_t)nbytes);
    if (!copy) {
      sqlite3_result_error_nomem(ctx);
      return;
    }
    memcpy(copy, blob, (size_t)nbytes);
    sqlite3_result_blob64(ctx, copy, (sqlite3_uint64)nbytes, free);
    sqlite3_result_subtype(ctx, SVE_ELEM_INT8);
    return;
  }
  if (vt == SQLITE_TEXT) {
    const char *txt = (const char *)sqlite3_value_text(argv[0]);
    int tlen = sqlite3_value_bytes(argv[0]);
    if (!txt || tlen <= 0 || (size_t)tlen > SVE_MAX_JSON_BYTES) {
      sqlite3_result_error(ctx, "invalid int8 JSON vector", -1);
      return;
    }
    int8_t *iv = NULL;
    size_t dims = 0U;
    int rc = sve_parse_i8_json(txt, (size_t)tlen, SVE_MAX_TABLE_DIMS, &iv, &dims);
    if (rc != SVE_OK) {
      scalar_error(ctx, rc == SVE_ERR_NOMEM ? SQLITE_NOMEM : SQLITE_ERROR,
                   NULL, "invalid or out-of-range int8 JSON");
      return;
    }
    sqlite3_result_blob64(ctx, iv, (sqlite3_uint64)dims, free);
    sqlite3_result_subtype(ctx, SVE_ELEM_INT8);
    return;
  }
  sqlite3_result_error(ctx, "sve_int8 requires BLOB or JSON TEXT", -1);
}

static void fn_sve_bit(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  (void)argc;
  if (sqlite3_value_type(argv[0]) != SQLITE_BLOB) {
    sqlite3_result_error(ctx, "sve_bit requires BLOB input", -1);
    return;
  }
  int nbytes = sqlite3_value_bytes(argv[0]);
  const void *blob = sqlite3_value_blob(argv[0]);
  if (!blob || nbytes <= 0 || (size_t)nbytes > SVE_MAX_SCALAR_DIMS / 8U) {
    sqlite3_result_error(ctx, "invalid bitvector BLOB", -1);
    return;
  }
  void *copy = malloc((size_t)nbytes);
  if (!copy) {
    sqlite3_result_error_nomem(ctx);
    return;
  }
  memcpy(copy, blob, (size_t)nbytes);
  sqlite3_result_blob64(ctx, copy, (sqlite3_uint64)nbytes, free);
  sqlite3_result_subtype(ctx, SVE_ELEM_BIT);
}

static void fn_sve_length(sqlite3_context *ctx, int argc,
                          sqlite3_value **argv) {
  (void)argc;
  char *err = NULL;
  scalar_vec_t v;
  int rc = extract_scalar_vec(argv[0], &v, &err);
  if (rc != SQLITE_OK) {
    scalar_error(ctx, rc, err, "invalid vector");
    return;
  }
  sqlite3_result_int64(ctx, (sqlite3_int64)v.dims);
  free_scalar_vec(&v);
}

static void fn_sve_to_json(sqlite3_context *ctx, int argc,
                           sqlite3_value **argv) {
  (void)argc;
  char *err = NULL;
  scalar_vec_t v;
  int rc = extract_scalar_vec(argv[0], &v, &err);
  if (rc != SQLITE_OK) {
    scalar_error(ctx, rc, err, "invalid vector");
    return;
  }
  sqlite3_str *s = sqlite3_str_new(sqlite3_context_db_handle(ctx));
  sqlite3_str_appendchar(s, 1, '[');
  if (v.elem_type == SVE_ELEM_FLOAT32) {
    const float *fp = (const float *)v.data;
    for (size_t i = 0U; i < v.dims && sqlite3_str_errcode(s) == SQLITE_OK; i++) {
      sqlite3_str_appendf(s, i > 0U ? ",%.9g" : "%.9g", (double)fp[i]);
    }
  } else if (v.elem_type == SVE_ELEM_INT8) {
    const int8_t *ip = (const int8_t *)v.data;
    for (size_t i = 0U; i < v.dims && sqlite3_str_errcode(s) == SQLITE_OK; i++) {
      sqlite3_str_appendf(s, i > 0U ? ",%d" : "%d", (int)ip[i]);
    }
  } else {
    free_scalar_vec(&v);
    sqlite3_free(sqlite3_str_finish(s));
    sqlite3_result_error(ctx, "sve_to_json does not support bitvectors", -1);
    return;
  }
  sqlite3_str_appendchar(s, 1, ']');
  rc = sqlite3_str_errcode(s);
  char *out = sqlite3_str_finish(s);
  free_scalar_vec(&v);
  if (rc != SQLITE_OK || !out) {
    sqlite3_free(out);
    if (rc == SQLITE_TOOBIG) {
      sqlite3_result_error_toobig(ctx);
    } else {
      sqlite3_result_error_nomem(ctx);
    }
    return;
  }
  sqlite3_result_text(ctx, out, -1, sqlite3_free);
}

static void fn_sve_distance_l2(sqlite3_context *ctx, int argc,
                               sqlite3_value **argv) {
  (void)argc;
  char *err = NULL;
  scalar_vec_t a, b;
  int rc = extract_matching_pair(argv[0], argv[1], &a, &b, &err);
  if (rc != SQLITE_OK) {
    scalar_error(ctx, rc, err, "invalid arguments");
    return;
  }
  if (a.elem_type == SVE_ELEM_BIT) {
    free_scalar_vec(&a);
    free_scalar_vec(&b);
    sqlite3_result_error(ctx, "cannot compute L2 distance on bitvectors", -1);
    return;
  }
  double dist = 0.0;
  int drc = sve_compute_distance(a.elem_type, SVE_METRIC_L2, a.data, b.data,
                                 a.dims, &dist);
  free_scalar_vec(&a);
  free_scalar_vec(&b);
  if (drc != SVE_OK) {
    sqlite3_result_error(ctx, "L2 distance computation failed or overflowed",
                         -1);
    return;
  }
  sqlite3_result_double(ctx, dist);
}

static void fn_sve_distance_l1(sqlite3_context *ctx, int argc,
                               sqlite3_value **argv) {
  (void)argc;
  char *err = NULL;
  scalar_vec_t a, b;
  int rc = extract_matching_pair(argv[0], argv[1], &a, &b, &err);
  if (rc != SQLITE_OK) {
    scalar_error(ctx, rc, err, "invalid arguments");
    return;
  }
  if (a.elem_type == SVE_ELEM_INT8) {
    int64_t dist_i64 = 0;
    int drc = sve_distance_l1_i8((const int8_t *)a.data, (const int8_t *)b.data,
                                 a.dims, &dist_i64);
    free_scalar_vec(&a);
    free_scalar_vec(&b);
    if (drc != SVE_OK) {
      sqlite3_result_error(ctx, "int8 L1 computation failed", -1);
      return;
    }
    sqlite3_result_int64(ctx, (sqlite3_int64)dist_i64);
    return;
  }
  if (a.elem_type == SVE_ELEM_BIT) {
    free_scalar_vec(&a);
    free_scalar_vec(&b);
    sqlite3_result_error(ctx, "cannot compute L1 distance on bitvectors", -1);
    return;
  }
  double dist = 0.0;
  int drc = sve_distance_l1_f32((const float *)a.data, (const float *)b.data,
                                a.dims, &dist);
  free_scalar_vec(&a);
  free_scalar_vec(&b);
  if (drc != SVE_OK) {
    sqlite3_result_error(ctx, "float32 L1 computation failed", -1);
    return;
  }
  sqlite3_result_double(ctx, dist);
}

static void fn_sve_distance_cosine(sqlite3_context *ctx, int argc,
                                   sqlite3_value **argv) {
  (void)argc;
  char *err = NULL;
  scalar_vec_t a, b;
  int rc = extract_matching_pair(argv[0], argv[1], &a, &b, &err);
  if (rc != SQLITE_OK) {
    scalar_error(ctx, rc, err, "invalid arguments");
    return;
  }
  if (a.elem_type == SVE_ELEM_BIT) {
    free_scalar_vec(&a);
    free_scalar_vec(&b);
    sqlite3_result_error(ctx, "cannot compute cosine distance on bitvectors",
                         -1);
    return;
  }
  double dist = 0.0;
  int drc = sve_compute_distance(a.elem_type, SVE_METRIC_COSINE, a.data, b.data,
                                 a.dims, &dist);
  free_scalar_vec(&a);
  free_scalar_vec(&b);
  if (drc != SVE_OK) {
    sqlite3_result_error(ctx, "cosine computation failed", -1);
    return;
  }
  sqlite3_result_double(ctx, dist);
}

static void fn_sve_distance_ip(sqlite3_context *ctx, int argc,
                               sqlite3_value **argv) {
  (void)argc;
  char *err = NULL;
  scalar_vec_t a, b;
  int rc = extract_matching_pair(argv[0], argv[1], &a, &b, &err);
  if (rc != SQLITE_OK) {
    scalar_error(ctx, rc, err, "invalid arguments");
    return;
  }
  if (a.elem_type != SVE_ELEM_FLOAT32) {
    free_scalar_vec(&a);
    free_scalar_vec(&b);
    sqlite3_result_error(ctx, "sve_distance_ip requires float32 vectors", -1);
    return;
  }
  double dist = 0.0;
  int drc =
      sve_distance_ip_f32((const float *)a.data, (const float *)b.data, a.dims,
                          &dist);
  free_scalar_vec(&a);
  free_scalar_vec(&b);
  if (drc != SVE_OK) {
    sqlite3_result_error(ctx, "inner product overflow or invalid input", -1);
    return;
  }
  sqlite3_result_double(ctx, dist);
}

static void fn_sve_distance_hamming(sqlite3_context *ctx, int argc,
                                    sqlite3_value **argv) {
  (void)argc;
  char *err = NULL;
  scalar_vec_t a, b;
  int rc = extract_matching_pair(argv[0], argv[1], &a, &b, &err);
  if (rc != SQLITE_OK) {
    scalar_error(ctx, rc, err, "invalid arguments");
    return;
  }
  if (a.elem_type != SVE_ELEM_BIT) {
    free_scalar_vec(&a);
    free_scalar_vec(&b);
    sqlite3_result_error(ctx, "sve_distance_hamming requires bitvectors", -1);
    return;
  }
  uint64_t h = 0U;
  int drc = sve_distance_hamming((const uint8_t *)a.data, (const uint8_t *)b.data,
                                 a.dims, &h);
  free_scalar_vec(&a);
  free_scalar_vec(&b);
  if (drc != SVE_OK) {
    sqlite3_result_error(ctx, "hamming computation failed", -1);
    return;
  }
  sqlite3_result_int64(ctx, (sqlite3_int64)h);
}

static void fn_sve_arith(sqlite3_context *ctx, sqlite3_value **argv, bool is_sub,
                         bool saturate) {
  char *err = NULL;
  scalar_vec_t a, b;
  int rc = extract_matching_pair(argv[0], argv[1], &a, &b, &err);
  if (rc != SQLITE_OK) {
    scalar_error(ctx, rc, err, "invalid arguments");
    return;
  }
  if (a.elem_type == SVE_ELEM_BIT) {
    free_scalar_vec(&a);
    free_scalar_vec(&b);
    sqlite3_result_error(ctx, "bitvector arithmetic is not supported", -1);
    return;
  }
  if (a.elem_type == SVE_ELEM_FLOAT32) {
    size_t nbytes = a.dims * sizeof(float);
    float *out = (float *)malloc(nbytes);
    if (!out) {
      free_scalar_vec(&a);
      free_scalar_vec(&b);
      sqlite3_result_error_nomem(ctx);
      return;
    }
    int arc = is_sub
                  ? sve_sub_f32((const float *)a.data, (const float *)b.data,
                                out, a.dims)
                  : sve_add_f32((const float *)a.data, (const float *)b.data,
                                out, a.dims);
    free_scalar_vec(&a);
    free_scalar_vec(&b);
    if (arc != SVE_OK) {
      free(out);
      sqlite3_result_error(ctx, "float32 arithmetic overflow to non-finite",
                           -1);
      return;
    }
    sqlite3_result_blob64(ctx, out, (sqlite3_uint64)nbytes, free);
    sqlite3_result_subtype(ctx, SVE_ELEM_FLOAT32);
    return;
  }

  int8_t *out = (int8_t *)malloc(a.dims);
  if (!out) {
    free_scalar_vec(&a);
    free_scalar_vec(&b);
    sqlite3_result_error_nomem(ctx);
    return;
  }
  size_t out_dims = a.dims;
  int arc = is_sub
                ? sve_sub_i8((const int8_t *)a.data, (const int8_t *)b.data,
                             out, out_dims, saturate)
                : sve_add_i8((const int8_t *)a.data, (const int8_t *)b.data,
                             out, out_dims, saturate);
  free_scalar_vec(&a);
  free_scalar_vec(&b);
  if (arc != SVE_OK) {
    free(out);
    sqlite3_result_error(ctx, "int8 vector arithmetic overflow [-128..127]", -1);
    return;
  }
  sqlite3_result_blob64(ctx, out, (sqlite3_uint64)out_dims, free);
  sqlite3_result_subtype(ctx, SVE_ELEM_INT8);
}

static void fn_sve_add(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  (void)argc;
  fn_sve_arith(ctx, argv, false, false);
}

static void fn_sve_add_sat(sqlite3_context *ctx, int argc,
                           sqlite3_value **argv) {
  (void)argc;
  fn_sve_arith(ctx, argv, false, true);
}

static void fn_sve_sub(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  (void)argc;
  fn_sve_arith(ctx, argv, true, false);
}

static void fn_sve_quantize_int8(sqlite3_context *ctx, int argc,
                                 sqlite3_value **argv) {
  (void)argc;
  char *err = NULL;
  scalar_vec_t v;
  int rc = extract_scalar_vec(argv[0], &v, &err);
  if (rc != SQLITE_OK) {
    scalar_error(ctx, rc, err, "invalid vector");
    return;
  }
  if (v.elem_type != SVE_ELEM_FLOAT32) {
    free_scalar_vec(&v);
    sqlite3_result_error(ctx, "sve_quantize_int8 requires float32 input", -1);
    return;
  }
  int8_t *q = (int8_t *)malloc(v.dims);
  if (!q) {
    free_scalar_vec(&v);
    sqlite3_result_error_nomem(ctx);
    return;
  }
  float scale = 0.0f;
  int qrc = sve_quantize_f32_to_sq8((const float *)v.data, v.dims, q, &scale);
  size_t dims = v.dims;
  free_scalar_vec(&v);
  if (qrc != SVE_OK) {
    free(q);
    sqlite3_result_error(ctx, "SQ8 quantization failed", -1);
    return;
  }
  sqlite3_result_blob64(ctx, q, (sqlite3_uint64)dims, free);
  sqlite3_result_subtype(ctx, SVE_ELEM_INT8);
}

static void fn_sve_redact(sqlite3_context *ctx, int argc,
                          sqlite3_value **argv) {
  (void)argc;
  if (sqlite3_value_type(argv[0]) == SQLITE_NULL) {
    sqlite3_result_null(ctx);
    return;
  }
  const char *txt = (const char *)sqlite3_value_text(argv[0]);
  int tlen = sqlite3_value_bytes(argv[0]);
  if (!txt || tlen < 0 || (size_t)tlen > SVE_MAX_TEXT_BYTES) {
    sqlite3_result_error(ctx, "input text exceeds maximum allowed length", -1);
    return;
  }
  size_t cap = (size_t)tlen * 2U + 64U;
  char *out = (char *)malloc(cap);
  if (!out) {
    sqlite3_result_error_nomem(ctx);
    return;
  }
  size_t redactions = 0U;
  if (sve_privacy_scrub(txt, (size_t)tlen, out, cap, &redactions) != SVE_OK) {
    free(out);
    sqlite3_result_error(ctx, "privacy scrub failed", -1);
    return;
  }
  sqlite3_result_text(ctx, out, -1, free);
}

int sve_hybrid_register(sqlite3 *db, char **pzErrMsg) {
  if (!db) {
    return SQLITE_MISUSE;
  }
#ifndef SQLITE_CORE
  if (!sqlite3_api) {
    return SQLITE_MISUSE;
  }
  if (sqlite3_libversion_number() < 3045000) {
    if (pzErrMsg) {
      *pzErrMsg = sqlite3_mprintf("sve0 requires SQLite 3.45.0 or later");
    }
    return SQLITE_ERROR;
  }
#endif
  const int k_base = SQLITE_UTF8 | SQLITE_INNOCUOUS | SQLITE_DETERMINISTIC;
  static const struct {
    const char *name;
    int narg;
    int flags;
    void (*fn)(sqlite3_context *, int, sqlite3_value **);
  } k_funcs[] = {
      {"sve_version", 0, SQLITE_UTF8 | SQLITE_INNOCUOUS | SQLITE_DETERMINISTIC,
       fn_sve_version},
      {"sve_f32", 1, SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, fn_sve_f32},
      {"sve_int8", 1, SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, fn_sve_int8},
      {"sve_bit", 1, SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, fn_sve_bit},
      {"sve_length", 1, SQLITE_SUBTYPE, fn_sve_length},
      {"sve_to_json", 1, SQLITE_SUBTYPE, fn_sve_to_json},
      {"sve_distance_l2", 2, SQLITE_SUBTYPE, fn_sve_distance_l2},
      {"sve_distance_l1", 2, SQLITE_SUBTYPE, fn_sve_distance_l1},
      {"sve_distance_cosine", 2, SQLITE_SUBTYPE, fn_sve_distance_cosine},
      {"sve_distance_ip", 2, SQLITE_SUBTYPE, fn_sve_distance_ip},
      {"sve_distance_hamming", 2, SQLITE_SUBTYPE, fn_sve_distance_hamming},
      {"sve_add", 2, SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, fn_sve_add},
      {"sve_add_sat", 2, SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE,
       fn_sve_add_sat},
      {"sve_sub", 2, SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE, fn_sve_sub},
      {"sve_quantize_int8", 1, SQLITE_SUBTYPE | SQLITE_RESULT_SUBTYPE,
       fn_sve_quantize_int8},
      {"sve_redact", 1, 0, fn_sve_redact},
  };

  sqlite3_mutex *mutex = sqlite3_db_mutex(db);
  sqlite3_mutex_enter(mutex);
  /* Removing a registered function is forbidden while a VM is running. */
  for (sqlite3_stmt *st = sqlite3_next_stmt(db, NULL); st;
       st = sqlite3_next_stmt(db, st)) {
    if (sqlite3_stmt_busy(st)) {
      if (pzErrMsg) {
        *pzErrMsg = sqlite3_mprintf("load sve0 through the C API or .load with no active statements");
      }
      sqlite3_mutex_leave(mutex);
      return SQLITE_BUSY;
    }
  }

  size_t installed = 0U;
  int rc = SQLITE_OK;
  for (; installed < sizeof(k_funcs) / sizeof(k_funcs[0]); installed++) {
    rc = sqlite3_create_function_v2(
        db, k_funcs[installed].name, k_funcs[installed].narg,
        k_base | k_funcs[installed].flags, NULL,
        k_funcs[installed].fn, NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
      if (pzErrMsg) {
        *pzErrMsg = sqlite3_mprintf("failed to register %s: %s",
                                    k_funcs[installed].name, sqlite3_errmsg(db));
      }
      break;
    }
  }
  if (rc == SQLITE_OK) {
    rc = sve_register_vtab_module(db, pzErrMsg);
  }
  if (rc != SQLITE_OK) {
    while (installed > 0U) {
      installed--;
      sqlite3_create_function_v2(db, k_funcs[installed].name,
                                k_funcs[installed].narg, SQLITE_UTF8,
                                NULL, NULL, NULL, NULL, NULL);
    }
  }
  sqlite3_mutex_leave(mutex);
  return rc;
}

SVE_HYBRID_API int sqlite3_svehybrid_init(sqlite3 *db, char **pzErrMsg,
                                          const sqlite3_api_routines *pApi) {
#ifndef SQLITE_CORE
  SQLITE_EXTENSION_INIT2(pApi);
#else
  (void)pApi;
#endif
  return sve_hybrid_register(db, pzErrMsg);
}

SVE_HYBRID_API int sqlite3_vechybrid_init(sqlite3 *db, char **pzErrMsg,
                                          const sqlite3_api_routines *pApi) {
  return sqlite3_svehybrid_init(db, pzErrMsg, pApi);
}
