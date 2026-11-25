#include "sve_hybrid/sve_vtab_guard.h"
#include "sve_hybrid/sve_hnsw.h"
#include "sve_hybrid/sve_hybrid_rank.h"
#include "sve_hybrid/sve_numeric.h"

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SQLITE_CORE
SQLITE_EXTENSION_INIT3
#endif

#define SVE_UNUSED(x) ((void)(x))

/* -------------------------------------------------------------------------- */
/* Privacy & OOD Secret Scrubber                                              */
/* -------------------------------------------------------------------------- */

static bool is_token_char(unsigned char c) {
  return (isalnum(c) != 0) || c == '_' || c == '-' || c == '.' || c == '+' ||
         c == '/' || c == '=' || c == '~';
}

static bool is_id_digit(unsigned char c) { return (isdigit(c) != 0); }

static bool ci_starts_with(const char *s, size_t rem, const char *lit) {
  size_t n = strlen(lit);
  if (rem < n) {
    return false;
  }
  for (size_t i = 0; i < n; i++) {
    if (tolower((unsigned char)s[i]) != tolower((unsigned char)lit[i])) {
      return false;
    }
  }
  return true;
}

static size_t match_pem_block(const char *s, size_t rem) {
  if (!ci_starts_with(s, rem, "-----BEGIN ")) {
    return 0U;
  }
  static const char *const labels[] = {
      "PRIVATE KEY", "RSA PRIVATE KEY", "EC PRIVATE KEY", "DSA PRIVATE KEY",
      "ENCRYPTED PRIVATE KEY", "OPENSSH PRIVATE KEY", "PGP PRIVATE KEY BLOCK"
  };
  for (size_t k = 0U; k < sizeof(labels) / sizeof(labels[0]); k++) {
    size_t n = strlen(labels[k]);
    size_t header = 11U + n + 5U;
    if (rem < header || !ci_starts_with(s + 11U, rem - 11U, labels[k]) ||
        memcmp(s + 11U + n, "-----", 5U) != 0) {
      continue;
    }
    size_t footer = 9U + n + 5U;
    for (size_t i = header; i + footer <= rem; i++) {
      if (ci_starts_with(s + i, rem - i, "-----END ") &&
          ci_starts_with(s + i + 9U, rem - i - 9U, labels[k]) &&
          memcmp(s + i + 9U + n, "-----", 5U) == 0) {
        return i + footer;
      }
    }
    return rem;
  }
  return 0U;
}

static size_t match_prefixed_token(const char *s, size_t rem, size_t pos,
                                   const char *full_buf) {
  if (pos > 0U && is_token_char((unsigned char)full_buf[pos - 1U]) &&
      full_buf[pos - 1U] != ':' && full_buf[pos - 1U] != '=' &&
      full_buf[pos - 1U] != '.') {
    return 0U;
  }
  static const struct {
    const char *prefix;
    size_t min_tail;
  } k_prefixes[] = {
      {"sk-proj-", 10U}, {"sk-live-", 10U}, {"sk-", 14U},
      {"ghp_", 12U},     {"gho_", 12U},     {"ghu_", 12U},
      {"AKIA", 12U},     {"xoxb-", 10U},    {"xoxp-", 10U},
      {"AIza", 16U},
  };
  for (size_t k = 0U; k < sizeof(k_prefixes) / sizeof(k_prefixes[0]); k++) {
    size_t plen = strlen(k_prefixes[k].prefix);
    if (rem >= plen + k_prefixes[k].min_tail &&
        strncmp(s, k_prefixes[k].prefix, plen) == 0) {
      size_t j = plen;
      while (j < rem && is_token_char((unsigned char)s[j]) && s[j] != '.') {
        j++;
      }
      if (j >= plen + k_prefixes[k].min_tail) {
        return j;
      }
    }
  }
  return 0U;
}

static size_t match_bearer_token(const char *s, size_t rem, size_t pos,
                                 const char *full_buf) {
  if (pos > 0U && isalnum((unsigned char)full_buf[pos - 1U])) {
    return 0U;
  }
  if (!ci_starts_with(s, rem, "Bearer ")) {
    return 0U;
  }
  size_t i = 7U;
  while (i < rem && (s[i] == ' ' || s[i] == '\t')) {
    i++;
  }
  size_t start = i;
  bool has_digit = false, has_sym = false, has_upper = false, has_lower = false;
  while (i < rem && is_token_char((unsigned char)s[i])) {
    unsigned char c = (unsigned char)s[i];
    if (isdigit(c)) {
      has_digit = true;
    } else if (isupper(c)) {
      has_upper = true;
    } else if (islower(c)) {
      has_lower = true;
    } else {
      has_sym = true;
    }
    i++;
  }
  size_t tok_len = i - start;
  /* Require non-prose secret characteristics so "Bearer authentication" stays untouched */
  if (tok_len >= 12U &&
      (has_digit || has_sym || (has_upper && has_lower))) {
    return i;
  }
  return 0U;
}

static size_t match_dsn_credential(const char *s, size_t rem,
                                     size_t *out_secret_start,
                                     size_t *out_secret_end) {
  if (rem < 6U || strncmp(s, "://", 3U) != 0) {
    return 0U;
  }
  size_t colon = 0U;
  size_t at_pos = 0U;
  for (size_t i = 3U; i < rem; i++) {
    char c = s[i];
    if (c == '/' || c == '?' || c == '#' || c == '"' || c == '\'' ||
        c == '<' || c == '>' || isspace((unsigned char)c)) {
      return 0U;
    }
    if (c == ':' && colon == 0U) {
      colon = i;
    } else if (c == '@') {
      at_pos = i;
      break;
    }
  }
  if (colon > 3U && at_pos > colon + 1U) {
    *out_secret_start = colon + 1U;
    *out_secret_end = at_pos;
    return at_pos;
  }
  return 0U;
}

static size_t match_labeled_secret_or_id(const char *s, size_t rem, size_t pos,
                                         const char *full_buf,
                                         size_t *out_val_start,
                                         size_t *out_val_end) {
  if (pos > 0U && (isalnum((unsigned char)full_buf[pos - 1U]) ||
                   full_buf[pos - 1U] == '_')) {
    return 0U;
  }
  static const char *const k_labels[] = {
      "api_key",       "apikey",      "secret_key",   "access_token",
      "auth_token",    "password",    "passwd",       "quic_ticket",
      "session_ticket","imsi",        "imei",         "msisdn",
      "phone",         "mobile",
  };
  for (size_t k = 0U; k < sizeof(k_labels) / sizeof(k_labels[0]); k++) {
    size_t llen = strlen(k_labels[k]);
    if (!ci_starts_with(s, rem, k_labels[k])) {
      continue;
    }
    size_t i = llen;
    while (i < rem && (s[i] == ' ' || s[i] == '\t')) {
      i++;
    }
    if (i >= rem || (s[i] != ':' && s[i] != '=')) {
      continue;
    }
    i++;
    while (i < rem && (s[i] == ' ' || s[i] == '\t')) {
      i++;
    }
    char quote = 0;
    if (i < rem && (s[i] == '"' || s[i] == '\'')) {
      quote = s[i++];
    }
    size_t vstart = i;
    if (quote != 0) {
      while (i < rem && s[i] != quote && s[i] != '\r' && s[i] != '\n') {
        if (s[i] == '\\' && i + 1U < rem &&
            s[i + 1U] != '\r' && s[i + 1U] != '\n') {
          i++;
        }
        i++;
      }
    } else {
      while (i < rem && s[i] != ',' && s[i] != ';' && s[i] != '"' &&
             s[i] != '\'' && s[i] != '<' && s[i] != '>' &&
             !isspace((unsigned char)s[i])) {
        i++;
      }
    }
    size_t minimum = k < 9U ? 1U : 6U;
    if (i - vstart >= minimum) {
      *out_val_start = vstart;
      *out_val_end = i;
      return i;
    }
  }
  return 0U;
}

static bool context_before_is_benign_id(const char *buf, size_t pos) {
  size_t back = pos > 24U ? pos - 24U : 0U;
  size_t len = pos - back;
  static const char *const k_benign[] = {
      "order", "invoice", "ticket_no", "batch", "sku", "tx",
      "sha256", "hash", "uuid", "trace", "id=", "id:"};
  for (size_t k = 0U; k < sizeof(k_benign) / sizeof(k_benign[0]); k++) {
    size_t kl = strlen(k_benign[k]);
    for (size_t j = 0U; j + kl <= len; j++) {
      if (ci_starts_with(buf + back + j, len - j, k_benign[k])) {
        return true;
      }
    }
  }
  return false;
}

static size_t match_phone_literal(const char *s, size_t rem, size_t pos,
                                  const char *full_buf) {
  /* +CC-XXX... international format */
  if (rem >= 9U && s[0] == '+' && is_id_digit((unsigned char)s[1])) {
    size_t digits = 0U;
    size_t i = 1U;
    while (i < rem &&
           (is_id_digit((unsigned char)s[i]) || s[i] == '-' || s[i] == ' ')) {
      if (is_id_digit((unsigned char)s[i])) {
        digits++;
      }
      i++;
    }
    while (i > 1U && (s[i - 1U] == '-' || s[i - 1U] == ' ')) {
      i--;
    }
    if (digits >= 8U && digits <= 15U) {
      return i;
    }
  }
  /* Standalone 1[3-9]xxxxxxxxx Chinese mobile not inside longer digits/ID context */
  if (pos > 0U && (is_id_digit((unsigned char)full_buf[pos - 1U]) ||
                   isalpha((unsigned char)full_buf[pos - 1U]) ||
                   full_buf[pos - 1U] == '-')) {
    return 0U;
  }
  if (rem >= 11U && s[0] == '1' && s[1] >= '3' && s[1] <= '9') {
    bool all_digits = true;
    for (size_t i = 0U; i < 11U; i++) {
      if (!is_id_digit((unsigned char)s[i])) {
        all_digits = false;
        break;
      }
    }
    bool trail_ok = (rem == 11U) || (!is_id_digit((unsigned char)s[11]) &&
                                     !isalpha((unsigned char)s[11]) &&
                                     s[11] != '-');
    if (all_digits && trail_ok && !context_before_is_benign_id(full_buf, pos)) {
      return 11U;
    }
  }
  return 0U;
}

static int append_slice(char *out, size_t cap, size_t *pos, const char *src,
                        size_t len) {
  if (len >= cap - *pos) {
    return SVE_ERR_LIMIT;
  }
  if (len > 0U) {
    memcpy(out + *pos, src, len);
    *pos += len;
  }
  out[*pos] = '\0';
  return SVE_OK;
}

int sve_privacy_scrub(const char *input, size_t input_len, char *out_buf,
                      size_t out_cap, size_t *out_redactions) {
  if (!out_buf || out_cap == 0U || (!input && input_len > 0U) ||
      input_len > SVE_MAX_TEXT_BYTES) {
    return SVE_ERR_INVALID;
  }
  out_buf[0] = '\0';
  if (out_redactions) {
    *out_redactions = 0U;
  }
  if (input_len == 0U) {
    return SVE_OK;
  }

  if (memchr(input, 0, input_len) != NULL) {
    return SVE_ERR_INVALID;
  }

  static const char k_tag[] = "[REDACTED]";
  static const size_t k_tag_len = sizeof(k_tag) - 1U;
  size_t out_pos = 0U;
  size_t redactions = 0U;
  size_t i = 0U;

  while (i < input_len) {
    size_t rem = input_len - i;
    const char *s = input + i;
    size_t m = match_pem_block(s, rem);
    if (m > 0U) {
      if (append_slice(out_buf, out_cap, &out_pos, k_tag, k_tag_len) != SVE_OK) {
        out_buf[0] = '\0';
        return SVE_ERR_LIMIT;
      }
      redactions++;
      i += m;
      continue;
    }

    size_t sec_s = 0U, sec_e = 0U;
    m = match_dsn_credential(s, rem, &sec_s, &sec_e);
    if (m > 0U) {
      if (append_slice(out_buf, out_cap, &out_pos, s, sec_s) != SVE_OK ||
          append_slice(out_buf, out_cap, &out_pos, k_tag, k_tag_len) !=
              SVE_OK) {
        out_buf[0] = '\0';
        return SVE_ERR_LIMIT;
      }
      redactions++;
      i += sec_e;
      continue;
    }

    size_t val_s = 0U, val_e = 0U;
    m = match_labeled_secret_or_id(s, rem, i, input, &val_s, &val_e);
    if (m > 0U) {
      if (append_slice(out_buf, out_cap, &out_pos, s, val_s) != SVE_OK ||
          append_slice(out_buf, out_cap, &out_pos, k_tag, k_tag_len) !=
              SVE_OK) {
        out_buf[0] = '\0';
        return SVE_ERR_LIMIT;
      }
      redactions++;
      i += val_e;
      continue;
    }

    m = match_bearer_token(s, rem, i, input);
    if (m > 0U) {
      if (append_slice(out_buf, out_cap, &out_pos, "Bearer ", 7U) != SVE_OK ||
          append_slice(out_buf, out_cap, &out_pos, k_tag, k_tag_len) !=
              SVE_OK) {
        out_buf[0] = '\0';
        return SVE_ERR_LIMIT;
      }
      redactions++;
      i += m;
      continue;
    }

    m = match_prefixed_token(s, rem, i, input);
    if (m > 0U) {
      if (append_slice(out_buf, out_cap, &out_pos, k_tag, k_tag_len) != SVE_OK) {
        out_buf[0] = '\0';
        return SVE_ERR_LIMIT;
      }
      redactions++;
      i += m;
      continue;
    }

    m = match_phone_literal(s, rem, i, input);
    if (m > 0U) {
      if (append_slice(out_buf, out_cap, &out_pos, k_tag, k_tag_len) != SVE_OK) {
        out_buf[0] = '\0';
        return SVE_ERR_LIMIT;
      }
      redactions++;
      i += m;
      continue;
    }

    if (append_slice(out_buf, out_cap, &out_pos, s, 1U) != SVE_OK) {
      out_buf[0] = '\0';
      return SVE_ERR_LIMIT;
    }
    i++;
  }

  if (out_redactions) {
    *out_redactions = redactions;
  }
  return SVE_OK;
}

/* -------------------------------------------------------------------------- */
/* Vector Value Coercion (#361 bound untyped BLOB fix)                        */
/* -------------------------------------------------------------------------- */

int sve_coerce_vector_value(sqlite3_value *val, sve_elem_type_t expected_type,
                            size_t expected_dims, void **out_buf,
                            size_t *out_bytes, char **pzErr) {
  if (!val || !out_buf || !out_bytes || expected_dims == 0U ||
      expected_dims > SVE_MAX_TABLE_DIMS) {
    if (pzErr) {
      *pzErr = sqlite3_mprintf("invalid vector specification");
    }
    return SQLITE_ERROR;
  }
  *out_buf = NULL;
  *out_bytes = 0U;

  int vtype = sqlite3_value_type(val);
  int subtype = sqlite3_value_subtype(val);
  size_t need_bytes = sve_elem_byte_size(expected_type, expected_dims);

  if (vtype == SQLITE_BLOB) {
    if (subtype != 0 && subtype != (int)expected_type) {
      if (pzErr) {
        *pzErr = sqlite3_mprintf("vector element subtype mismatch: got %d, "
                                 "expected %d",
                                 subtype, (int)expected_type);
      }
      return SQLITE_ERROR;
    }
    int nbytes = sqlite3_value_bytes(val);
    if (nbytes <= 0 || (size_t)nbytes != need_bytes) {
      if (pzErr) {
        *pzErr = sqlite3_mprintf("vector BLOB byte size mismatch: got %d, "
                                 "expected %llu",
                                 nbytes, (sqlite3_uint64)need_bytes);
      }
      return SQLITE_ERROR;
    }
    const void *src = sqlite3_value_blob(val);
    if (!src) {
      if (pzErr) {
        *pzErr = sqlite3_mprintf("null vector BLOB");
      }
      return SQLITE_ERROR;
    }
    void *copy = malloc(need_bytes);
    if (!copy) {
      return SQLITE_NOMEM;
    }
    memcpy(copy, src, need_bytes);
    if (expected_type == SVE_ELEM_FLOAT32 &&
        !sve_f32_array_is_finite((const float *)copy, expected_dims)) {
      free(copy);
      if (pzErr) {
        *pzErr = sqlite3_mprintf("float32 vector BLOB contains NaN or Inf");
      }
      return SQLITE_ERROR;
    }
    *out_buf = copy;
    *out_bytes = need_bytes;
    return SQLITE_OK;
  }

  if (vtype == SQLITE_TEXT) {
    const char *txt = (const char *)sqlite3_value_text(val);
    int tlen = sqlite3_value_bytes(val);
    if (!txt || tlen <= 0 || (size_t)tlen > SVE_MAX_JSON_BYTES) {
      if (pzErr) {
        *pzErr = sqlite3_mprintf("invalid or oversized JSON vector string");
      }
      return SQLITE_ERROR;
    }
    if (expected_type == SVE_ELEM_FLOAT32) {
      float *fv = NULL;
      size_t dims = 0U;
      int rc = sve_parse_f32_json(txt, (size_t)tlen, SVE_MAX_TABLE_DIMS, &fv,
                                  &dims);
      if (rc == SVE_ERR_NOMEM) {
        return SQLITE_NOMEM;
      }
      if (rc != SVE_OK || dims != expected_dims) {
        free(fv);
        if (pzErr) {
          *pzErr = sqlite3_mprintf("float32 JSON dimension/syntax error "
                                   "(expected %llu, got %llu)",
                                   (sqlite3_uint64)expected_dims,
                                   (sqlite3_uint64)dims);
        }
        return SQLITE_ERROR;
      }
      *out_buf = fv;
      *out_bytes = need_bytes;
      return SQLITE_OK;
    }
    if (expected_type == SVE_ELEM_INT8) {
      int8_t *iv = NULL;
      size_t dims = 0U;
      int rc =
          sve_parse_i8_json(txt, (size_t)tlen, SVE_MAX_TABLE_DIMS, &iv, &dims);
      if (rc == SVE_ERR_NOMEM) {
        return SQLITE_NOMEM;
      }
      if (rc != SVE_OK || dims != expected_dims) {
        free(iv);
        if (pzErr) {
          *pzErr = sqlite3_mprintf("int8 JSON dimension/syntax error "
                                   "(expected %llu, got %llu)",
                                   (sqlite3_uint64)expected_dims,
                                   (sqlite3_uint64)dims);
        }
        return SQLITE_ERROR;
      }
      *out_buf = iv;
      *out_bytes = need_bytes;
      return SQLITE_OK;
    }
    if (pzErr) {
      *pzErr = sqlite3_mprintf("JSON input not supported for bit columns");
    }
    return SQLITE_ERROR;
  }

  if (pzErr) {
    *pzErr = sqlite3_mprintf("vector value must be BLOB or JSON TEXT");
  }
  return SQLITE_ERROR;
}

/* -------------------------------------------------------------------------- */
/* `sve0` SQLite Hardened Hybrid Virtual Table                                */
/* -------------------------------------------------------------------------- */

enum {
  SVE_COL_ID = 0,
  SVE_COL_VEC = 1,
  SVE_COL_USER_BASE = 2
};

typedef struct sve_user_col {
  char name[64];
  int sql_type; /* SQLITE_INTEGER, SQLITE_FLOAT, SQLITE_TEXT */
  bool is_fulltext;
} sve_user_col_t;

typedef struct sve_vtab {
  sqlite3_vtab base;
  sqlite3 *db;
  char *schema_name;
  char *table_name;

  char vec_col_name[64];
  sve_elem_type_t elem_type;
  sve_metric_t metric;
  sve_index_kind_t index_kind;
  size_t dims;
  uint32_t hnsw_m;
  uint32_t hnsw_ef_c;
  uint32_t hnsw_ef_s;
  bool redact_text;

  sve_user_col_t user_cols[SVE_MAX_USER_COLS];
  size_t n_user_cols;
  int fulltext_col_idx; /* -1 if none */

  int col_distance;
  int col_k;
  int col_text_query;
  int col_hybrid_alpha;
  int col_rrf_score;

  struct sve_cursor *match_cursor;
  int match_column;
  sve_hnsw_t *hnsw;
  bool hnsw_dirty;
  bool hnsw_rebuild;
  bool hnsw_version_valid;
  sqlite3_int64 hnsw_version;
} sve_vtab_t;

typedef struct sve_result_row {
  int64_t rowid;
  double distance;
  double fused_score;
} sve_result_row_t;

typedef struct sve_cursor {
  sqlite3_vtab_cursor base;
  int plan; /* 1=fullscan, 2=point, 3=knn, 4=hybrid */
  sqlite3_stmt *scan_stmt;
  bool scan_eof;

  sve_result_row_t *hits;
  size_t n_hits;
  size_t cursor_idx;
  sqlite3_stmt *row_fetch_stmt;
  sqlite3_value *vector_query;
  sqlite3_value *text_query;
  size_t query_k;
  double query_alpha;
  bool has_alpha;
} sve_cursor_t;

enum {
  SVE_PLAN_FULLSCAN = 1,
  SVE_PLAN_POINT = 2,
  SVE_PLAN_KNN = 3,
  SVE_PLAN_HYBRID = 4,
  SVE_PLAN_MASK = 7,
  SVE_PLAN_K = 8,
  SVE_PLAN_ALPHA = 16,
  SVE_PLAN_RECHECK_MATCH = 32
};

static char *dup_str(const char *s) {
  if (!s) {
    return NULL;
  }
  size_t n = strlen(s);
  char *p = (char *)malloc(n + 1U);
  if (p) {
    memcpy(p, s, n + 1U);
  }
  return p;
}

static char *trim_copy(const char *s, size_t len) {
  while (len > 0U && isspace((unsigned char)s[0])) {
    s++;
    len--;
  }
  while (len > 0U && isspace((unsigned char)s[len - 1U])) {
    len--;
  }
  char *p = (char *)malloc(len + 1U);
  if (p) {
    if (len > 0U) {
      memcpy(p, s, len);
    }
    p[len] = '\0';
  }
  return p;
}

static int parse_u32_strict(const char *s, uint32_t *out) {
  if (!s || !isdigit((unsigned char)s[0])) {
    return -1;
  }
  char *endp = NULL;
  errno = 0;
  unsigned long v = strtoul(s, &endp, 10);
  if (errno == ERANGE || *endp != '\0' || v > UINT32_MAX) {
    return -1;
  }
  *out = (uint32_t)v;
  return 0;
}

static bool copy_column_name(char *name, size_t capacity, const char *token) {
  size_t n = 0U;
  char quote = token[0];
  if (quote == '[' || quote == '"' || quote == '\'' || quote == '`') {
    char end = quote == '[' ? ']' : quote;
    token++;
    for (;;) {
      char c = *token++;
      if (c == '\0') {
        return false;
      }
      if (c == end) {
        if (quote == '[' || *token != end) {
          if (*token != '\0') {
            return false;
          }
          break;
        }
        token++;
      }
      if (n == capacity - 1U) {
        return false;
      }
      name[n++] = c;
    }
  } else {
    n = strlen(token);
    if (n >= capacity) {
      return false;
    }
    memcpy(name, token, n);
  }
  name[n] = '\0';
  return n != 0U;
}

static int parse_vector_def(const char *arg, sve_vtab_t *p, char **pzErr) {
  /* Format: "<name> float|int8|bit[dims] [index=...] [metric=...] [m=...] ..." */
  char *copy = dup_str(arg);
  if (!copy) {
    return SQLITE_NOMEM;
  }
  size_t len = strlen(copy);
  char *toks[16];
  size_t nt = 0U;
  size_t pos = 0U;
  while (pos < len && nt < 16U) {
    while (pos < len && isspace((unsigned char)copy[pos])) {
      pos++;
    }
    if (pos >= len) {
      break;
    }
    toks[nt++] = copy + pos;
    while (pos < len && !isspace((unsigned char)copy[pos])) {
      pos++;
    }
    if (pos < len) {
      copy[pos++] = '\0';
    }
  }

  if (pos < len || nt < 2U ||
      !copy_column_name(p->vec_col_name, sizeof(p->vec_col_name), toks[0])) {
    free(copy);
    *pzErr = sqlite3_mprintf("invalid vector column definition: %s", arg);
    return SQLITE_ERROR;
  }

  /* Parse type[dims] */
  const char *td = toks[1];
  const char *lb = strchr(td, '[');
  const char *rb = lb ? strchr(lb, ']') : NULL;
  if (!lb || !rb || rb[1] != '\0') {
    free(copy);
    *pzErr = sqlite3_mprintf("vector column must look like float[D]");
    return SQLITE_ERROR;
  }
  size_t tlen = (size_t)(lb - td);
  if (tlen == 5U && ci_starts_with(td, tlen, "float")) {
    p->elem_type = SVE_ELEM_FLOAT32;
  } else if (tlen == 7U && ci_starts_with(td, tlen, "float32")) {
    p->elem_type = SVE_ELEM_FLOAT32;
  } else if (tlen == 4U && ci_starts_with(td, tlen, "int8")) {
    p->elem_type = SVE_ELEM_INT8;
  } else if (tlen == 3U && ci_starts_with(td, tlen, "bit")) {
    p->elem_type = SVE_ELEM_BIT;
  } else {
    free(copy);
    *pzErr = sqlite3_mprintf("unsupported vector type");
    return SQLITE_ERROR;
  }

  char dim_buf[16];
  size_t dlen = (size_t)(rb - lb - 1);
  if (dlen == 0U || dlen >= sizeof(dim_buf)) {
    free(copy);
    *pzErr = sqlite3_mprintf("invalid vector dimensions");
    return SQLITE_ERROR;
  }
  memcpy(dim_buf, lb + 1, dlen);
  dim_buf[dlen] = '\0';
  uint32_t dims_u32 = 0U;
  if (parse_u32_strict(dim_buf, &dims_u32) != 0 || dims_u32 == 0U ||
      dims_u32 > SVE_MAX_TABLE_DIMS) {
    free(copy);
    *pzErr = sqlite3_mprintf("vector dimensions out of range [1..%u]",
                             SVE_MAX_TABLE_DIMS);
    return SQLITE_ERROR;
  }
  p->dims = (size_t)dims_u32;

  for (size_t i = 2U; i < nt; i++) {
    const char *kv = toks[i];
    if (sqlite3_stricmp(kv, "index=flat") == 0) {
      p->index_kind = SVE_INDEX_FLAT;
    } else if (sqlite3_stricmp(kv, "index=hnsw_sq8") == 0) {
      p->index_kind = SVE_INDEX_HNSW_SQ8;
    } else if (sqlite3_stricmp(kv, "index=hnsw") == 0) {
      p->index_kind = SVE_INDEX_HNSW;
    } else if (sqlite3_stricmp(kv, "metric=l2") == 0) {
      p->metric = SVE_METRIC_L2;
    } else if (sqlite3_stricmp(kv, "metric=cosine") == 0) {
      p->metric = SVE_METRIC_COSINE;
    } else if (sqlite3_stricmp(kv, "metric=l1") == 0) {
      p->metric = SVE_METRIC_L1;
    } else if (sqlite3_stricmp(kv, "metric=ip") == 0) {
      p->metric = SVE_METRIC_IP;
    } else if (sqlite3_stricmp(kv, "metric=hamming") == 0) {
      p->metric = SVE_METRIC_HAMMING;
    } else if (ci_starts_with(kv, strlen(kv), "m=")) {
      if (parse_u32_strict(kv + 2, &p->hnsw_m) != 0 || p->hnsw_m < 2U ||
          p->hnsw_m > SVE_MAX_HNSW_M) {
        free(copy);
        *pzErr = sqlite3_mprintf("invalid HNSW M parameter");
        return SQLITE_ERROR;
      }
    } else if (ci_starts_with(kv, strlen(kv), "ef_construction=")) {
      if (parse_u32_strict(kv + 16, &p->hnsw_ef_c) != 0 ||
          p->hnsw_ef_c < 4U || p->hnsw_ef_c > SVE_MAX_HNSW_EF) {
        free(copy);
        *pzErr = sqlite3_mprintf("invalid ef_construction parameter");
        return SQLITE_ERROR;
      }
    } else if (ci_starts_with(kv, strlen(kv), "ef_search=")) {
      if (parse_u32_strict(kv + 10, &p->hnsw_ef_s) != 0 ||
          p->hnsw_ef_s < 1U || p->hnsw_ef_s > SVE_MAX_HNSW_EF) {
        free(copy);
        *pzErr = sqlite3_mprintf("invalid ef_search parameter");
        return SQLITE_ERROR;
      }
    } else {
      free(copy);
      *pzErr = sqlite3_mprintf("unknown vector option");
      return SQLITE_ERROR;
    }
  }

  free(copy);
  return SQLITE_OK;
}

static int parse_user_col_def(const char *arg, sve_vtab_t *p, char **pzErr) {
  char *trimmed = trim_copy(arg, strlen(arg));
  if (!trimmed) {
    return SQLITE_NOMEM;
  }
  if (strcmp(trimmed, "redact=1") == 0 || strcmp(trimmed, "privacy=1") == 0) {
    p->redact_text = true;
    free(trimmed);
    return SQLITE_OK;
  }
  if (strcmp(trimmed, "redact=0") == 0 || strcmp(trimmed, "privacy=0") == 0) {
    p->redact_text = false;
    free(trimmed);
    return SQLITE_OK;
  }

  if (p->n_user_cols >= SVE_MAX_USER_COLS) {
    free(trimmed);
    *pzErr = sqlite3_mprintf("too many metadata columns (max %u)",
                             SVE_MAX_USER_COLS);
    return SQLITE_ERROR;
  }

  char *toks[4] = {0};
  size_t nt = 0U;
  size_t pos = 0U;
  size_t len = strlen(trimmed);
  while (pos < len && nt < 4U) {
    while (pos < len && isspace((unsigned char)trimmed[pos])) {
      pos++;
    }
    if (pos >= len) {
      break;
    }
    toks[nt++] = trimmed + pos;
    while (pos < len && !isspace((unsigned char)trimmed[pos])) {
      pos++;
    }
    if (pos < len) {
      trimmed[pos++] = '\0';
    }
  }
  if (nt < 2U || nt > 3U) {
    free(trimmed);
    *pzErr = sqlite3_mprintf("invalid column argument: %s", arg);
    return SQLITE_ERROR;
  }
  const char *cname = toks[0];
  if (cname[0] == '+') {
    cname++;
  }
  sve_user_col_t *uc = &p->user_cols[p->n_user_cols];
  memset(uc, 0, sizeof(*uc));
  if (!copy_column_name(uc->name, sizeof(uc->name), cname)) {
    free(trimmed);
    *pzErr = sqlite3_mprintf("invalid column name in: %s", arg);
    return SQLITE_ERROR;
  }

  if (sqlite3_stricmp(toks[1], "int") == 0 ||
      sqlite3_stricmp(toks[1], "integer") == 0) {
    uc->sql_type = SQLITE_INTEGER;
  } else if (sqlite3_stricmp(toks[1], "real") == 0 ||
             sqlite3_stricmp(toks[1], "float") == 0) {
    uc->sql_type = SQLITE_FLOAT;
  } else if (sqlite3_stricmp(toks[1], "text") == 0) {
    uc->sql_type = SQLITE_TEXT;
  } else {
    free(trimmed);
    *pzErr = sqlite3_mprintf("unsupported metadata column type");
    return SQLITE_ERROR;
  }

  if (nt == 3U) {
    if (sqlite3_stricmp(toks[2], "fulltext") != 0) {
      free(trimmed);
      *pzErr = sqlite3_mprintf("unknown metadata column option");
      return SQLITE_ERROR;
    }
    if (uc->sql_type != SQLITE_TEXT || p->fulltext_col_idx >= 0) {
      free(trimmed);
      *pzErr = sqlite3_mprintf("at most one TEXT column may be marked fulltext");
      return SQLITE_ERROR;
    }
    uc->is_fulltext = true;
    p->fulltext_col_idx = (int)p->n_user_cols;
  }

  p->n_user_cols++;
  free(trimmed);
  return SQLITE_OK;
}

static const char *sql_type_decl(int sql_type) {
  switch (sql_type) {
  case SQLITE_INTEGER:
    return "INTEGER";
  case SQLITE_FLOAT:
    return "REAL";
  default:
    return "TEXT";
  }
}

static sve_hnsw_config_t table_index_config(const sve_vtab_t *p) {
  sve_hnsw_config_t cfg =
      sve_hnsw_default_config(p->dims, p->elem_type, p->metric, p->index_kind);
  cfg.m = p->hnsw_m;
  cfg.ef_construction = p->hnsw_ef_c;
  cfg.ef_search = p->hnsw_ef_s;
  return cfg;
}

static int sqlite_index_error(int status) {
  switch (status) {
  case SVE_OK:
    return SQLITE_OK;
  case SVE_ERR_NOMEM:
    return SQLITE_NOMEM;
  case SVE_ERR_LIMIT:
    return SQLITE_TOOBIG;
  case SVE_ERR_CORRUPT:
  case SVE_ERR_NOTFOUND:
    return SQLITE_CORRUPT;
  default:
    return SQLITE_ERROR;
  }
}

static int rebuild_hnsw_from_rows(sve_vtab_t *p) {
  sve_hnsw_config_t cfg = table_index_config(p);
  sve_hnsw_t *rebuilt = sve_hnsw_create(&cfg);
  if (!rebuilt) {
    return SQLITE_NOMEM;
  }

  char *sql = sqlite3_mprintf(
      "SELECT id, vec FROM \"%w\".\"%w_rows\" ORDER BY id;",
      p->schema_name, p->table_name);
  if (!sql) {
    sve_hnsw_destroy(rebuilt);
    return SQLITE_NOMEM;
  }
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2(p->db, sql, -1, &stmt, NULL);
  sqlite3_free(sql);
  size_t need_bytes = sve_elem_byte_size(p->elem_type, p->dims);
  void *vec = NULL;
  if (rc == SQLITE_OK) {
    vec = malloc(need_bytes);
    if (!vec) {
      rc = SQLITE_NOMEM;
    }
  }
  if (rc == SQLITE_OK) {
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      if (sqlite3_column_type(stmt, 1) != SQLITE_BLOB) {
        rc = SQLITE_CORRUPT;
        break;
      }
      const void *blob = sqlite3_column_blob(stmt, 1);
      if (!blob || (size_t)sqlite3_column_bytes(stmt, 1) != need_bytes) {
        rc = SQLITE_CORRUPT;
        break;
      }
      memcpy(vec, blob, need_bytes);
      int status = sve_hnsw_insert(rebuilt, sqlite3_column_int64(stmt, 0), vec);
      if (status != SVE_OK) {
        rc = status == SVE_ERR_INVALID ? SQLITE_CORRUPT : sqlite_index_error(status);
        break;
      }
    }
  }
  sqlite3_finalize(stmt);
  free(vec);
  if (rc != SQLITE_DONE) {
    sve_hnsw_destroy(rebuilt);
    return rc;
  }
  sve_hnsw_destroy(p->hnsw);
  p->hnsw = rebuilt;
  p->hnsw_rebuild = false;
  return SQLITE_OK;
}

static int save_hnsw_snapshot(sve_vtab_t *p) {
  uint8_t *buf = NULL;
  size_t len = 0U;
  int rc = sqlite_index_error(sve_hnsw_serialize(p->hnsw, &buf, &len));
  if (rc != SQLITE_OK) {
    return rc;
  }

  char *sql = sqlite3_mprintf(
      "INSERT OR REPLACE INTO \"%w\".\"%w_info\"(key, val) VALUES "
      "('hnsw_snapshot', ?1);", p->schema_name, p->table_name);
  if (!sql) {
    free(buf);
    return SQLITE_NOMEM;
  }
  sqlite3_stmt *stmt = NULL;
  rc = sqlite3_prepare_v2(p->db, sql, -1, &stmt, NULL);
  sqlite3_free(sql);
  if (rc == SQLITE_OK) {
    rc = sqlite3_bind_blob64(stmt, 1, buf, (sqlite3_uint64)len, SQLITE_STATIC);
  }
  if (rc == SQLITE_OK) {
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_DONE) {
      rc = sqlite3_changes64(p->db) == 1 ? SQLITE_OK : SQLITE_ERROR;
    }
  }
  sqlite3_finalize(stmt);
  free(buf);
  return rc;
}

static int load_hnsw_from_snapshot(sve_vtab_t *p, char **pzErr) {
  char *sql = sqlite3_mprintf(
      "SELECT val FROM \"%w\".\"%w_info\" WHERE key='hnsw_snapshot';",
      p->schema_name, p->table_name);
  if (!sql) {
    return SQLITE_NOMEM;
  }
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2(p->db, sql, -1, &stmt, NULL);
  sqlite3_free(sql);
  if (rc == SQLITE_OK) {
    rc = sqlite3_step(stmt);
  }
  if (rc == SQLITE_DONE) {
    sqlite3_finalize(stmt);
    return rebuild_hnsw_from_rows(p);
  }
  if (rc == SQLITE_ROW) {
    rc = SQLITE_CORRUPT;
    if (sqlite3_column_type(stmt, 0) == SQLITE_BLOB) {
      const uint8_t *blob = (const uint8_t *)sqlite3_column_blob(stmt, 0);
      int bsize = sqlite3_column_bytes(stmt, 0);
      sve_hnsw_t *loaded = NULL;
      sve_hnsw_config_t cfg = table_index_config(p);
      rc = sqlite_index_error(sve_hnsw_deserialize_for_config(
          blob, (size_t)bsize, &cfg, &loaded));
      if (rc == SQLITE_OK) {
        sve_hnsw_destroy(p->hnsw);
        p->hnsw = loaded;
        p->hnsw_rebuild = false;
      }
    }
    if (rc == SQLITE_CORRUPT && pzErr) {
      sqlite3_free(*pzErr);
      *pzErr = sqlite3_mprintf("corrupt or incompatible HNSW snapshot");
    }
  }
  sqlite3_finalize(stmt);
  return rc;
}

static int refresh_hnsw(sve_vtab_t *p, char **pzErr) {
  if (p->hnsw_rebuild) {
    return rebuild_hnsw_from_rows(p);
  }
  if (p->hnsw_dirty) {
    return SQLITE_OK;
  }
  /* Pin the schema's read transaction while checking its pager version and
     loading the snapshot, so both reads see the same committed state. */
  char *sql = sqlite3_mprintf("PRAGMA \"%w\".data_version", p->schema_name);
  if (!sql) {
    return SQLITE_NOMEM;
  }
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2(p->db, sql, -1, &stmt, NULL);
  sqlite3_free(sql);
  if (rc == SQLITE_OK) {
    rc = sqlite3_step(stmt);
  }
  if (rc == SQLITE_ROW) {
    sqlite3_int64 version = sqlite3_column_int64(stmt, 0);
    rc = SQLITE_OK;
    if (!p->hnsw_version_valid || p->hnsw_version != version) {
      rc = load_hnsw_from_snapshot(p, pzErr);
      if (rc == SQLITE_OK) {
        p->hnsw_version = version;
        p->hnsw_version_valid = true;
      }
    }
  } else if (rc == SQLITE_DONE) {
    rc = SQLITE_ERROR;
  }
  sqlite3_finalize(stmt);
  return rc;
}

static int declare_schema(sve_vtab_t *p) {
  sqlite3_str *decl = sqlite3_str_new(p->db);
  sqlite3_str_appendf(decl, "CREATE TABLE x(\"%w\" BLOB", p->vec_col_name);
  for (size_t i = 0U; i < p->n_user_cols; i++) {
    sqlite3_str_appendf(decl, ", \"%w\" %s", p->user_cols[i].name,
                        sql_type_decl(p->user_cols[i].sql_type));
  }
  sqlite3_str_appendall(
      decl,
      ", distance REAL HIDDEN, k INTEGER HIDDEN, text_query TEXT HIDDEN, "
      "hybrid_alpha REAL HIDDEN, rrf_score REAL HIDDEN)");
  char *sql = sqlite3_str_finish(decl);
  if (!sql) {
    return SQLITE_NOMEM;
  }
  int rc = sqlite3_declare_vtab(p->db, sql);
  sqlite3_free(sql);
  p->col_distance = 1 + (int)p->n_user_cols;
  p->col_k = p->col_distance + 1;
  p->col_text_query = p->col_distance + 2;
  p->col_hybrid_alpha = p->col_distance + 3;
  p->col_rrf_score = p->col_distance + 4;
  return rc;
}

static int sve_vtab_init_common(sqlite3 *db, int argc, const char *const *argv,
                                bool is_create, sqlite3_vtab **ppVTab,
                                char **pzErr) {
  if (argc < 4) {
    *pzErr = sqlite3_mprintf("sve0 requires a vector column definition");
    return SQLITE_ERROR;
  }
  sve_vtab_t *p = (sve_vtab_t *)calloc(1U, sizeof(sve_vtab_t));
  if (!p) {
    return SQLITE_NOMEM;
  }
  p->db = db;
  p->schema_name = dup_str(argv[1]);
  p->table_name = dup_str(argv[2]);
  p->elem_type = SVE_ELEM_FLOAT32;
  p->metric = SVE_METRIC_L2;
  p->index_kind = SVE_INDEX_HNSW;
  p->hnsw_m = SVE_DEFAULT_HNSW_M;
  p->hnsw_ef_c = SVE_DEFAULT_HNSW_EF_CONSTRUCTION;
  p->hnsw_ef_s = SVE_DEFAULT_HNSW_EF_SEARCH;
  p->fulltext_col_idx = -1;

  int rc = (!p->schema_name || !p->table_name)
               ? SQLITE_NOMEM : parse_vector_def(argv[3], p, pzErr);
  for (int i = 4; rc == SQLITE_OK && i < argc; i++) {
    rc = parse_user_col_def(argv[i], p, pzErr);
  }
  sve_hnsw_config_t cfg = table_index_config(p);
  if (rc == SQLITE_OK && !sve_hnsw_config_is_valid(&cfg)) {
    *pzErr = sqlite3_mprintf("unsupported vector type, dimensions or metric");
    rc = SQLITE_ERROR;
  }
  if (rc == SQLITE_OK) {
    rc = sqlite3_vtab_config(db, SQLITE_VTAB_CONSTRAINT_SUPPORT, 1);
  }
  if (rc != SQLITE_OK) {
    free(p->schema_name);
    free(p->table_name);
    free(p);
    return rc;
  }

  rc = declare_schema(p);
  if (rc != SQLITE_OK) {
    free(p->schema_name);
    free(p->table_name);
    free(p);
    return rc;
  }

  if (is_create) {
    sqlite3_str *ddl = sqlite3_str_new(db);
    sqlite3_str_appendf(
        ddl,
        "CREATE TABLE \"%w\".\"%w_info\"("
        "key TEXT PRIMARY KEY, val BLOB);"
        "CREATE TABLE \"%w\".\"%w_rows\"("
        "id INTEGER PRIMARY KEY, vec BLOB NOT NULL",
        p->schema_name, p->table_name, p->schema_name, p->table_name);
    for (size_t i = 0U; i < p->n_user_cols; i++) {
      sqlite3_str_appendf(ddl, ", c%d %s", (int)i,
                          sql_type_decl(p->user_cols[i].sql_type));
    }
    sqlite3_str_appendall(ddl, ");");
    for (size_t i = 0U; i < p->n_user_cols; i++) {
      sqlite3_str_appendf(
          ddl,
          "CREATE INDEX \"%w\".\"%w_rows_c%d_idx\" "
          "ON \"%w_rows\"(c%d);",
          p->schema_name, p->table_name, (int)i, p->table_name, (int)i);
    }
    if (p->fulltext_col_idx >= 0) {
      sqlite3_str_appendf(
          ddl,
          "CREATE VIRTUAL TABLE \"%w\".\"%w_fts\" "
          "USING fts5(doc_text, tokenize='unicode61');",
          p->schema_name, p->table_name);
    }
    char *ddl_sql = sqlite3_str_finish(ddl);
    if (!ddl_sql) {
      free(p->schema_name);
      free(p->table_name);
      free(p);
      return SQLITE_NOMEM;
    }
    char *err = NULL;
    rc = sqlite3_exec(db, ddl_sql, NULL, NULL, &err);
    sqlite3_free(ddl_sql);
    if (rc != SQLITE_OK) {
      *pzErr = err;
      free(p->schema_name);
      free(p->table_name);
      free(p);
      return rc;
    }
    p->hnsw = sve_hnsw_create(&cfg);
    if (!p->hnsw) {
      free(p->schema_name);
      free(p->table_name);
      free(p);
      return SQLITE_NOMEM;
    }
    rc = save_hnsw_snapshot(p);
    if (rc != SQLITE_OK) {
      sve_hnsw_destroy(p->hnsw);
      free(p->schema_name);
      free(p->table_name);
      free(p);
      return rc;
    }
  } else {
    rc = refresh_hnsw(p, pzErr);
    if (rc != SQLITE_OK) {
      sve_hnsw_destroy(p->hnsw);
      free(p->schema_name);
      free(p->table_name);
      free(p);
      return rc;
    }
  }

  *ppVTab = &p->base;
  return SQLITE_OK;
}

static int sve_xCreate(sqlite3 *db, void *pAux, int argc,
                       const char *const *argv, sqlite3_vtab **ppVTab,
                       char **pzErr) {
  SVE_UNUSED(pAux);
  return sve_vtab_init_common(db, argc, argv, true, ppVTab, pzErr);
}

static int sve_xConnect(sqlite3 *db, void *pAux, int argc,
                        const char *const *argv, sqlite3_vtab **ppVTab,
                        char **pzErr) {
  SVE_UNUSED(pAux);
  return sve_vtab_init_common(db, argc, argv, false, ppVTab, pzErr);
}

static int sve_xDisconnect(sqlite3_vtab *pVTab) {
  sve_vtab_t *p = (sve_vtab_t *)pVTab;
  sve_hnsw_destroy(p->hnsw);
  free(p->schema_name);
  free(p->table_name);
  free(p);
  return SQLITE_OK;
}

static int sve_xDestroy(sqlite3_vtab *pVTab) {
  sve_vtab_t *p = (sve_vtab_t *)pVTab;
  sqlite3_str *s = sqlite3_str_new(p->db);
  sqlite3_str_appendf(s,
                      "DROP TABLE IF EXISTS \"%w\".\"%w_rows\";"
                      "DROP TABLE IF EXISTS \"%w\".\"%w_info\";",
                      p->schema_name, p->table_name, p->schema_name,
                      p->table_name);
  if (p->fulltext_col_idx >= 0) {
    sqlite3_str_appendf(s, "DROP TABLE IF EXISTS \"%w\".\"%w_fts\";",
                        p->schema_name, p->table_name);
  }
  char *sql = sqlite3_str_finish(s);
  if (!sql) {
    return SQLITE_NOMEM;
  }
  sqlite3_free(p->base.zErrMsg);
  int rc = sqlite3_exec(p->db, sql, NULL, NULL, &p->base.zErrMsg);
  sqlite3_free(sql);
  return rc == SQLITE_OK ? sve_xDisconnect(pVTab) : rc;
}

static int sve_xRename(sqlite3_vtab *pVTab, const char *name) {
  SVE_UNUSED(name);
  sqlite3_free(pVTab->zErrMsg);
  pVTab->zErrMsg = sqlite3_mprintf("sve0 does not support ALTER TABLE RENAME");
  return SQLITE_ERROR;
}

static int sve_xSync(sqlite3_vtab *pVTab) {
  sve_vtab_t *p = (sve_vtab_t *)pVTab;
  if (!p->hnsw_dirty) {
    return SQLITE_OK;
  }
  int rc = refresh_hnsw(p, NULL);
  return rc == SQLITE_OK ? save_hnsw_snapshot(p) : rc;
}

static int sve_xCommit(sqlite3_vtab *pVTab) {
  sve_vtab_t *p = (sve_vtab_t *)pVTab;
  p->hnsw_dirty = false;
  return SQLITE_OK;
}

static int sve_xBegin(sqlite3_vtab *pVTab) {
  SVE_UNUSED(pVTab);
  return SQLITE_OK;
}

static int sve_xRollback(sqlite3_vtab *pVTab) {
  sve_vtab_t *p = (sve_vtab_t *)pVTab;
  p->hnsw_dirty = false;
  p->hnsw_rebuild = false;
  p->hnsw_version_valid = false;
  return SQLITE_OK;
}

static int query_error(sqlite3_vtab *tab, const char *message) {
  sqlite3_free(tab->zErrMsg);
  tab->zErrMsg = sqlite3_mprintf("%s", message);
  return SQLITE_ERROR;
}

static const char *comparison_sql(unsigned char op) {
  switch (op) {
  case SQLITE_INDEX_CONSTRAINT_EQ: return "=";
  case SQLITE_INDEX_CONSTRAINT_GT: return ">";
  case SQLITE_INDEX_CONSTRAINT_GE: return ">=";
  case SQLITE_INDEX_CONSTRAINT_LT: return "<";
  case SQLITE_INDEX_CONSTRAINT_LE: return "<=";
  default: return NULL;
  }
}

static int sve_xBestIndex(sqlite3_vtab *pVTab, sqlite3_index_info *pInfo) {
  sve_vtab_t *p = (sve_vtab_t *)pVTab;
  int match = -1, k = -1, text = -1, alpha = -1, rowid = -1;
  bool unusable_control = false;
  bool duplicate_control = false;

  for (int i = 0; i < pInfo->nConstraint; i++) {
    const struct sqlite3_index_constraint *c = &pInfo->aConstraint[i];
    bool is_match = c->op == SQLITE_INDEX_CONSTRAINT_MATCH;
    bool is_eq = c->op == SQLITE_INDEX_CONSTRAINT_EQ;
    int *slot = NULL;
    if (c->iColumn == 0 && is_match) {
      slot = &match;
    } else if (c->iColumn == p->col_text_query && (is_match || is_eq)) {
      slot = &text;
    } else if (c->iColumn == p->col_k && is_eq) {
      slot = &k;
    } else if (c->iColumn == p->col_hybrid_alpha && is_eq) {
      slot = &alpha;
    } else if (is_match) {
      return query_error(pVTab, "MATCH requires the vector or text_query column");
    } else if (c->iColumn == -1 && is_eq && c->usable) {
      rowid = i;
    }
    if (slot) {
      if (!c->usable) {
        unusable_control = true;
      } else if (*slot >= 0) {
        duplicate_control = true;
      } else {
        *slot = i;
      }
    }
  }

  if (unusable_control) {
    return SQLITE_CONSTRAINT;
  }
  if (match < 0) {
    if (text >= 0 &&
        pInfo->aConstraint[text].op == SQLITE_INDEX_CONSTRAINT_MATCH) {
      return query_error(pVTab, "text_query MATCH requires vector MATCH");
    }
    if (rowid >= 0) {
      pInfo->aConstraintUsage[rowid].argvIndex = 1;
      pInfo->aConstraintUsage[rowid].omit = 1;
      pInfo->idxNum = SVE_PLAN_POINT;
      pInfo->estimatedCost = 1.0;
      pInfo->estimatedRows = 1;
    } else {
      pInfo->idxNum = SVE_PLAN_FULLSCAN;
      pInfo->estimatedCost = 100000.0;
      pInfo->estimatedRows = 1000;
    }
    return SQLITE_OK;
  }
  if (duplicate_control) {
    return query_error(pVTab, "duplicate sve0 MATCH or search control");
  }

  const int controls[] = {match, k, text, alpha};
  int arg = 1;
  for (size_t i = 0U; i < sizeof(controls) / sizeof(controls[0]); i++) {
    if (controls[i] >= 0) {
      pInfo->aConstraintUsage[controls[i]].argvIndex = arg++;
      pInfo->aConstraintUsage[controls[i]].omit = 1;
    }
  }

  /* Build the predicate from validated column indices and SQLite collations. */
  sqlite3_str *where = sqlite3_str_new(p->db);
  bool has_filter = false;
  for (int i = 0; i < pInfo->nConstraint; i++) {
    const struct sqlite3_index_constraint *c = &pInfo->aConstraint[i];
    const char *op = comparison_sql(c->op);
    if (!c->usable || !op || c->iColumn < -1 || c->iColumn == 0 ||
        c->iColumn >= p->col_distance ||
        (c->op == SQLITE_INDEX_CONSTRAINT_EQ && sqlite3_vtab_in(pInfo, i, -1))) {
      continue;
    }
    if (c->iColumn == -1) {
      sqlite3_str_appendf(where, " AND id %s ?", op);
    } else {
      const char *collation = sqlite3_vtab_collation(pInfo, i);
      sqlite3_str_appendf(where, " AND c%d COLLATE \"%w\" %s ?",
                          c->iColumn - 1, collation ? collation : "BINARY", op);
    }
    pInfo->aConstraintUsage[i].argvIndex = arg++;
    has_filter = true;
  }
  int rc = sqlite3_str_errcode(where);
  pInfo->idxStr = sqlite3_str_finish(where);
  pInfo->needToFreeIdxStr = pInfo->idxStr != NULL;
  if (rc != SQLITE_OK || (has_filter && !pInfo->idxStr)) {
    return rc == SQLITE_OK ? SQLITE_NOMEM : rc;
  }
  pInfo->idxNum = (text >= 0 ? SVE_PLAN_HYBRID : SVE_PLAN_KNN) |
                 (k >= 0 ? SVE_PLAN_K : 0) |
                 (alpha >= 0 ? SVE_PLAN_ALPHA : 0) |
                 (pInfo->nConstraint > 16 ? SVE_PLAN_RECHECK_MATCH : 0);
  pInfo->estimatedCost = 10.0;
  pInfo->estimatedRows = 10;
  return SQLITE_OK;
}

static int sve_xOpen(sqlite3_vtab *pVTab, sqlite3_vtab_cursor **ppCursor) {
  SVE_UNUSED(pVTab);
  sve_cursor_t *cur = (sve_cursor_t *)calloc(1U, sizeof(sve_cursor_t));
  if (!cur) {
    return SQLITE_NOMEM;
  }
  *ppCursor = &cur->base;
  return SQLITE_OK;
}

static void reset_cursor_state(sve_cursor_t *cur) {
  sve_vtab_t *p = (sve_vtab_t *)cur->base.pVtab;
  if (p->match_cursor == cur) {
    p->match_cursor = NULL;
  }
  sqlite3_value_free(cur->vector_query);
  sqlite3_value_free(cur->text_query);
  cur->vector_query = NULL;
  cur->text_query = NULL;
  cur->query_k = 0U;
  cur->has_alpha = false;
  if (cur->scan_stmt) {
    sqlite3_finalize(cur->scan_stmt);
    cur->scan_stmt = NULL;
  }
  if (cur->row_fetch_stmt) {
    sqlite3_finalize(cur->row_fetch_stmt);
    cur->row_fetch_stmt = NULL;
  }
  free(cur->hits);
  cur->hits = NULL;
  cur->n_hits = 0U;
  cur->cursor_idx = 0U;
  cur->scan_eof = false;
}

static int sve_xClose(sqlite3_vtab_cursor *pCur) {
  sve_cursor_t *cur = (sve_cursor_t *)pCur;
  reset_cursor_state(cur);
  free(cur);
  return SQLITE_OK;
}

static int prepare_row_fetch(sve_vtab_t *p, sve_cursor_t *cur) {
  sqlite3_str *s = sqlite3_str_new(p->db);
  sqlite3_str_appendall(s, "SELECT vec");
  for (size_t i = 0U; i < p->n_user_cols; i++) {
    sqlite3_str_appendf(s, ", c%d", (int)i);
  }
  sqlite3_str_appendf(s, " FROM \"%w\".\"%w_rows\" WHERE id = ?1;",
                      p->schema_name, p->table_name);
  char *sql = sqlite3_str_finish(s);
  if (!sql) {
    return SQLITE_NOMEM;
  }
  int rc = sqlite3_prepare_v2(p->db, sql, -1, &cur->row_fetch_stmt, NULL);
  sqlite3_free(sql);
  return rc;
}

static int step_next_hit_row(sve_cursor_t *cur) {
  while (cur->cursor_idx < cur->n_hits) {
    int64_t rid = cur->hits[cur->cursor_idx].rowid;
    int rc = sqlite3_reset(cur->row_fetch_stmt);
    if (rc == SQLITE_OK) {
      rc = sqlite3_bind_int64(cur->row_fetch_stmt, 1, rid);
    }
    if (rc == SQLITE_OK) {
      rc = sqlite3_step(cur->row_fetch_stmt);
    }
    if (rc == SQLITE_ROW) {
      return SQLITE_OK;
    }
    if (rc == SQLITE_DONE) {
      cur->cursor_idx++;
      continue;
    }
    return rc;
  }
  cur->scan_eof = true;
  return SQLITE_OK;
}

static int bind_filters(sqlite3_stmt *stmt, int first, int argc,
                        sqlite3_value **argv) {
  if (sqlite3_bind_parameter_count(stmt) != first - 1 + argc) {
    return SQLITE_ERROR;
  }
  for (int i = 0; i < argc; i++) {
    int rc = sqlite3_bind_value(stmt, first + i, argv[i]);
    if (rc != SQLITE_OK) {
      return rc;
    }
  }
  return SQLITE_OK;
}

static int collect_metadata_allowlist(sve_vtab_t *p, const char *filters,
                                      int argc, sqlite3_value **argv,
                                      int64_t **out_ids, size_t *out_n) {
  *out_ids = NULL;
  *out_n = 0U;
  if (!filters) {
    return argc == 0 ? SQLITE_OK : SQLITE_ERROR;
  }
  char *sql = sqlite3_mprintf(
      "SELECT id FROM \"%w\".\"%w_rows\" WHERE 1=1%s ORDER BY id;",
      p->schema_name, p->table_name, filters);
  if (!sql) {
    return SQLITE_NOMEM;
  }
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2(p->db, sql, -1, &stmt, NULL);
  sqlite3_free(sql);
  if (rc == SQLITE_OK) {
    rc = bind_filters(stmt, 1, argc, argv);
  }

  size_t cap = 0U;
  size_t n = 0U;
  int64_t *ids = NULL;
  if (rc == SQLITE_OK) {
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      if (n == cap) {
        if (cap > SIZE_MAX / (2U * sizeof(*ids))) {
          rc = SQLITE_TOOBIG;
          break;
        }
        cap = cap != 0U ? cap * 2U : 64U;
        int64_t *tmp = (int64_t *)realloc(ids, cap * sizeof(*ids));
        if (!tmp) {
          rc = SQLITE_NOMEM;
          break;
        }
        ids = tmp;
      }
      ids[n++] = sqlite3_column_int64(stmt, 0);
    }
  }
  sqlite3_finalize(stmt);
  if (rc != SQLITE_DONE) {
    free(ids);
    return rc;
  }
  *out_ids = ids;
  *out_n = n;
  return SQLITE_OK;
}

static int run_fts5_candidates(sve_vtab_t *p, const char *raw_query,
                               const char *filters, int argc,
                               sqlite3_value **argv, size_t limit,
                               sve_search_hit_t **out_hits, size_t *out_n) {
  *out_hits = NULL;
  *out_n = 0U;
  if (p->fulltext_col_idx < 0) {
    return SQLITE_OK;
  }

  char sanitized[4096];
  int status = sve_fts5_sanitize_query(raw_query, 32U, sanitized, sizeof(sanitized));
  if (status != SVE_OK) {
    return sqlite_index_error(status);
  }
  if (sanitized[0] == '\0') {
    return SQLITE_OK;
  }

  sqlite3_str *s = sqlite3_str_new(p->db);
  sqlite3_str_appendf(
      s, "SELECT rowid, bm25(\"%w_fts\") FROM \"%w\".\"%w_fts\" "
         "WHERE \"%w_fts\" MATCH ?1",
      p->table_name, p->schema_name, p->table_name, p->table_name);
  if (filters) {
    sqlite3_str_appendf(s, " AND rowid IN (SELECT id FROM \"%w\".\"%w_rows\""
                           " WHERE 1=1%s)", p->schema_name, p->table_name, filters);
  }
  sqlite3_str_appendf(s, " ORDER BY 2, rowid LIMIT %u;", (unsigned int)limit);
  int rc = sqlite3_str_errcode(s);
  char *sql = sqlite3_str_finish(s);
  sqlite3_stmt *stmt = NULL;
  if (rc == SQLITE_OK) {
    rc = sqlite3_prepare_v2(p->db, sql, -1, &stmt, NULL);
  }
  sqlite3_free(sql);
  if (rc == SQLITE_OK) {
    rc = sqlite3_bind_text(stmt, 1, sanitized, -1, SQLITE_STATIC);
  }
  if (rc == SQLITE_OK) {
    rc = bind_filters(stmt, 2, argc, argv);
  }
  sve_search_hit_t *hits = NULL;
  size_t n = 0U;
  if (rc == SQLITE_OK) {
    hits = (sve_search_hit_t *)calloc(limit, sizeof(*hits));
    if (!hits) {
      rc = SQLITE_NOMEM;
    } else {
      while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        hits[n].rowid = sqlite3_column_int64(stmt, 0);
        hits[n].score_or_dist = sqlite3_column_double(stmt, 1);
        n++;
      }
    }
  }
  sqlite3_finalize(stmt);
  if (rc != SQLITE_DONE) {
    free(hits);
    return rc;
  }
  *out_hits = hits;
  *out_n = n;
  return SQLITE_OK;
}

static int sve_xFilter(sqlite3_vtab_cursor *pCur, int idxNum, const char *idxStr,
                       int argc, sqlite3_value **argv) {
  sve_cursor_t *cur = (sve_cursor_t *)pCur;
  sve_vtab_t *p = (sve_vtab_t *)pCur->pVtab;
  reset_cursor_state(cur);
  cur->plan = idxNum & SVE_PLAN_MASK;

  if (cur->plan == SVE_PLAN_FULLSCAN || cur->plan == SVE_PLAN_POINT) {
    sqlite3_str *s = sqlite3_str_new(p->db);
    sqlite3_str_appendall(s, "SELECT id, vec");
    for (size_t i = 0U; i < p->n_user_cols; i++) {
      sqlite3_str_appendf(s, ", c%d", (int)i);
    }
    sqlite3_str_appendf(s, " FROM \"%w\".\"%w_rows\"", p->schema_name,
                        p->table_name);
    if (cur->plan == SVE_PLAN_POINT) {
      sqlite3_str_appendall(s, " WHERE id = ?1");
    } else {
      sqlite3_str_appendall(s, " ORDER BY id ASC");
    }
    char *sql = sqlite3_str_finish(s);
    if (!sql) {
      return SQLITE_NOMEM;
    }
    int rc = sqlite3_prepare_v2(p->db, sql, -1, &cur->scan_stmt, NULL);
    sqlite3_free(sql);
    if (rc != SQLITE_OK) {
      return rc;
    }
    if (cur->plan == SVE_PLAN_POINT) {
      if (argc != 1) {
        return SQLITE_ERROR;
      }
      rc = sqlite3_bind_value(cur->scan_stmt, 1, argv[0]);
      if (rc != SQLITE_OK) {
        return rc;
      }
    }
    rc = sqlite3_step(cur->scan_stmt);
    if (rc == SQLITE_DONE) {
      cur->scan_eof = true;
      return SQLITE_OK;
    }
    return (rc == SQLITE_ROW) ? SQLITE_OK : rc;
  }

  bool hybrid = cur->plan == SVE_PLAN_HYBRID;
  bool explicit_k = (idxNum & SVE_PLAN_K) != 0;
  bool explicit_alpha = (idxNum & SVE_PLAN_ALPHA) != 0;
  int controls = 1 + explicit_k + hybrid + explicit_alpha;
  if (argc < controls ||
      (cur->plan != SVE_PLAN_KNN && cur->plan != SVE_PLAN_HYBRID)) {
    return SQLITE_ERROR;
  }
  int next = 1;
  size_t k = 10U;
  if (explicit_k) {
    sqlite3_value *kv = argv[next++];
    if (sqlite3_value_numeric_type(kv) != SQLITE_INTEGER ||
        sqlite3_value_int64(kv) < 1 ||
        sqlite3_value_int64(kv) > (sqlite3_int64)SVE_MAX_K) {
      return query_error(&p->base, "k must be a positive in-range integer");
    }
    k = (size_t)sqlite3_value_int64(kv);
  }
  const char *text_q = NULL;
  if (hybrid) {
    sqlite3_value *text = argv[next++];
    int type = sqlite3_value_type(text);
    if (type != SQLITE_NULL) {
      if (type != SQLITE_TEXT) {
        return query_error(&p->base, "text_query must be text or NULL");
      }
      text_q = (const char *)sqlite3_value_text(text);
      int len = sqlite3_value_bytes(text);
      if (!text_q) {
        return SQLITE_NOMEM;
      }
      if ((size_t)len > SVE_MAX_TEXT_BYTES ||
          memchr(text_q, 0, (size_t)len) != NULL) {
        return query_error(&p->base, "invalid or oversized text_query");
      }
    }
  }
  double alpha = 0.0;
  if (explicit_alpha) {
    sqlite3_value *value = argv[next++];
    int type = sqlite3_value_numeric_type(value);
    alpha = sqlite3_value_double(value);
    if ((type != SQLITE_INTEGER && type != SQLITE_FLOAT) ||
        !isfinite(alpha) || alpha < 0.0 || alpha > 1.0) {
      return query_error(&p->base, "hybrid_alpha must be a finite number in [0,1]");
    }
  }
  if (hybrid && !text_q) {
    cur->scan_eof = true;
    return SQLITE_OK;
  }
  cur->query_k = k;
  cur->query_alpha = alpha;
  cur->has_alpha = explicit_alpha;
  if (hybrid) {
    cur->text_query = sqlite3_value_dup(argv[1 + explicit_k]);
    if (!cur->text_query) {
      return SQLITE_NOMEM;
    }
  }
  if ((idxNum & SVE_PLAN_RECHECK_MATCH) != 0) {
    cur->vector_query = sqlite3_value_dup(argv[0]);
    if (!cur->vector_query) {
      return SQLITE_NOMEM;
    }
  }
  int rc = refresh_hnsw(p, &p->base.zErrMsg);
  if (rc != SQLITE_OK) {
    return rc;
  }

  void *qbuf = NULL;
  size_t qbytes = 0U;
  char *cerr = NULL;
  rc = sve_coerce_vector_value(argv[0], p->elem_type, p->dims, &qbuf,
                                   &qbytes, &cerr);
  if (rc != SQLITE_OK) {
    sqlite3_free(p->base.zErrMsg);
    p->base.zErrMsg = cerr;
    return rc;
  }

  int64_t *allowed = NULL;
  size_t n_allowed = 0U;
  rc = collect_metadata_allowlist(p, idxStr, argc - controls, argv + controls,
                                  &allowed, &n_allowed);
  if (rc != SQLITE_OK || (idxStr && n_allowed == 0U)) {
    free(qbuf);
    free(allowed);
    cur->scan_eof = true;
    return rc;
  }

  if (hybrid) {
    size_t pool_k = k * 4U;
    if (pool_k < 40U) {
      pool_k = 40U;
    }
    if (pool_k > SVE_MAX_K) {
      pool_k = SVE_MAX_K;
    }
    sve_search_hit_t *vhits =
        (sve_search_hit_t *)calloc(pool_k, sizeof(sve_search_hit_t));
    size_t nv = 0U;
    if (!vhits) {
      free(allowed);
      free(qbuf);
      return SQLITE_NOMEM;
    }
    rc = sqlite_index_error(sve_hnsw_search_allowlist(
        p->hnsw, qbuf, pool_k, 0U, allowed, n_allowed, vhits, &nv));
    sve_search_hit_t *bhits = NULL;
    size_t nb = 0U;
    if (rc == SQLITE_OK) {
      rc = run_fts5_candidates(p, text_q, idxStr, argc - controls,
                               argv + controls, pool_k, &bhits, &nb);
    }
    sve_hybrid_hit_t *fused =
        (sve_hybrid_hit_t *)calloc(k, sizeof(sve_hybrid_hit_t));
    size_t nf = 0U;
    if (rc == SQLITE_OK && fused) {
      int frc = explicit_alpha
                    ? sve_fuse_convex(vhits, nv, bhits, nb, alpha, k, fused,
                                      &nf)
                    : sve_fuse_rrf(vhits, nv, bhits, nb, 60U, k, fused, &nf);
      rc = sqlite_index_error(frc);
    } else if (rc == SQLITE_OK) {
      rc = SQLITE_NOMEM;
    }

    if (rc == SQLITE_OK && nf > 0U) {
      cur->hits = (sve_result_row_t *)calloc(nf, sizeof(sve_result_row_t));
      if (!cur->hits) {
        rc = SQLITE_NOMEM;
      } else {
        cur->n_hits = nf;
        for (size_t i = 0U; i < nf; i++) {
          cur->hits[i].rowid = fused[i].rowid;
          cur->hits[i].distance = fused[i].vec_distance;
          cur->hits[i].fused_score = fused[i].fused_score;
        }
      }
    }
    free(fused);
    free(bhits);
    free(vhits);
  } else {
    sve_search_hit_t *vhits =
        (sve_search_hit_t *)calloc(k, sizeof(sve_search_hit_t));
    size_t nv = 0U;
    if (!vhits) {
      free(allowed);
      free(qbuf);
      return SQLITE_NOMEM;
    }
    rc = sqlite_index_error(sve_hnsw_search_allowlist(
        p->hnsw, qbuf, k, 0U, allowed, n_allowed, vhits, &nv));
    if (rc != SQLITE_OK) {
      free(vhits);
      free(allowed);
      free(qbuf);
      return rc;
    }
    if (nv > 0U) {
      cur->hits = (sve_result_row_t *)calloc(nv, sizeof(sve_result_row_t));
      if (!cur->hits) {
        free(vhits);
        free(allowed);
        free(qbuf);
        return SQLITE_NOMEM;
      }
      cur->n_hits = nv;
      for (size_t i = 0U; i < nv; i++) {
        cur->hits[i].rowid = vhits[i].rowid;
        cur->hits[i].distance = vhits[i].score_or_dist;
        cur->hits[i].fused_score = 0.0;
      }
    }
    free(vhits);
  }

  free(allowed);
  free(qbuf);
  if (rc != SQLITE_OK) {
    return rc;
  }

  if (cur->n_hits == 0U) {
    cur->scan_eof = true;
    return SQLITE_OK;
  }
  rc = prepare_row_fetch(p, cur);
  if (rc != SQLITE_OK) {
    return rc;
  }
  return step_next_hit_row(cur);
}

static int sve_xNext(sqlite3_vtab_cursor *pCur) {
  sve_cursor_t *cur = (sve_cursor_t *)pCur;
  sve_vtab_t *p = (sve_vtab_t *)pCur->pVtab;
  if (p->match_cursor == cur) {
    p->match_cursor = NULL;
  }
  if (cur->plan == SVE_PLAN_FULLSCAN || cur->plan == SVE_PLAN_POINT) {
    int rc = sqlite3_step(cur->scan_stmt);
    if (rc == SQLITE_DONE) {
      cur->scan_eof = true;
      return SQLITE_OK;
    }
    return (rc == SQLITE_ROW) ? SQLITE_OK : rc;
  }
  cur->cursor_idx++;
  return step_next_hit_row(cur);
}

static int sve_xEof(sqlite3_vtab_cursor *pCur) {
  sve_cursor_t *cur = (sve_cursor_t *)pCur;
  return cur->scan_eof ? 1 : 0;
}

static int sve_xColumn(sqlite3_vtab_cursor *pCur, sqlite3_context *ctx,
                       int iCol) {
  sve_cursor_t *cur = (sve_cursor_t *)pCur;
  sve_vtab_t *p = (sve_vtab_t *)pCur->pVtab;
  p->match_cursor = NULL;

  if (iCol >= p->col_distance && sqlite3_vtab_nochange(ctx)) {
    return SQLITE_OK;
  }
  if (iCol == p->col_distance) {
    if (cur->plan == SVE_PLAN_KNN || cur->plan == SVE_PLAN_HYBRID) {
      sqlite3_result_double(ctx, cur->hits[cur->cursor_idx].distance);
    } else {
      sqlite3_result_null(ctx);
    }
    return SQLITE_OK;
  }
  if (iCol == p->col_rrf_score) {
    if (cur->plan == SVE_PLAN_HYBRID) {
      sqlite3_result_double(ctx, cur->hits[cur->cursor_idx].fused_score);
    } else {
      sqlite3_result_null(ctx);
    }
    return SQLITE_OK;
  }
  if (iCol >= p->col_distance) {
    if (iCol == p->col_k && cur->query_k != 0U) {
      sqlite3_result_int64(ctx, (sqlite3_int64)cur->query_k);
    } else if (iCol == p->col_hybrid_alpha && cur->has_alpha) {
      sqlite3_result_double(ctx, cur->query_alpha);
    } else if (iCol == p->col_text_query && cur->text_query) {
      p->match_cursor = cur;
      p->match_column = iCol;
      sqlite3_result_value(ctx, cur->text_query);
    } else {
      sqlite3_result_null(ctx);
    }
    return SQLITE_OK;
  }

  sqlite3_stmt *st = (cur->plan == SVE_PLAN_FULLSCAN || cur->plan == SVE_PLAN_POINT)
                         ? cur->scan_stmt
                         : cur->row_fetch_stmt;
  int stmt_col = (cur->plan == SVE_PLAN_FULLSCAN || cur->plan == SVE_PLAN_POINT)
                     ? (iCol + 1) /* skip id column at 0 */
                     : iCol;      /* vec is 0, c0 is 1... */
  sqlite3_value *v = sqlite3_column_value(st, stmt_col);
  sqlite3_result_value(ctx, v);
  if (iCol == 0 && v && sqlite3_value_type(v) == SQLITE_BLOB) {
    sqlite3_result_subtype(ctx, (unsigned int)p->elem_type);
    if (cur->vector_query) {
      p->match_cursor = cur;
      p->match_column = iCol;
    }
  }
  return SQLITE_OK;
}

/* SQLite can re-evaluate MATCH even when xBestIndex sets omit. Confirm the
   argument against the search that produced the current row. */
static bool same_match_value(sqlite3_value *a, sqlite3_value *b) {
  if (!a || !b) {
    return false;
  }
  int type = sqlite3_value_type(a);
  if ((type != SQLITE_TEXT && type != SQLITE_BLOB) ||
      type != sqlite3_value_type(b)) {
    return false;
  }
  const void *left = type == SQLITE_TEXT ? sqlite3_value_text(a)
                                         : sqlite3_value_blob(a);
  const void *right = type == SQLITE_TEXT ? sqlite3_value_text(b)
                                          : sqlite3_value_blob(b);
  int bytes = sqlite3_value_bytes(a);
  return left && right && bytes == sqlite3_value_bytes(b) &&
         memcmp(left, right, (size_t)bytes) == 0;
}

static void recheck_match(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  SVE_UNUSED(argc);
  sve_vtab_t *p = (sve_vtab_t *)sqlite3_user_data(ctx);
  const sve_cursor_t *cur = p->match_cursor;
  p->match_cursor = NULL;
  if (cur) {
    bool vector = p->match_column == 0;
    sqlite3_value *original = vector ? cur->vector_query : cur->text_query;
    sqlite3_value *cell = vector ? sqlite3_column_value(cur->row_fetch_stmt, 0)
                                 : cur->text_query;
    if (same_match_value(original, argv[0]) && same_match_value(cell, argv[1])) {
      sqlite3_result_int(ctx, 1);
      return;
    }
  }
  sqlite3_result_error(ctx, "sve0 MATCH needs a matching search constraint", -1);
}

static int sve_xFindFunction(sqlite3_vtab *tab, int argc, const char *name,
                             void (**fn)(sqlite3_context *, int, sqlite3_value **),
                             void **arg) {
  if (argc == 2 && sqlite3_stricmp(name, "match") == 0) {
    *fn = recheck_match;
    *arg = tab;
    return 1;
  }
  return 0;
}

static int sve_xRowid(sqlite3_vtab_cursor *pCur, sqlite_int64 *pRowid) {
  sve_cursor_t *cur = (sve_cursor_t *)pCur;
  if (cur->plan == SVE_PLAN_FULLSCAN || cur->plan == SVE_PLAN_POINT) {
    *pRowid = sqlite3_column_int64(cur->scan_stmt, 0);
  } else {
    *pRowid = cur->hits[cur->cursor_idx].rowid;
  }
  return SQLITE_OK;
}

static int validate_strict_user_values(sve_vtab_t *p, int argc,
                                       sqlite3_value **argv,
                                       char **scrubbed_texts, char **pzErr) {
  for (size_t i = 0U; i < p->n_user_cols; i++) {
    int aidx = 2 + 1 + (int)i;
    if (aidx >= argc) {
      continue;
    }
    sqlite3_value *v = argv[aidx];
    int vt = sqlite3_value_type(v);
    if (vt == SQLITE_NULL) {
      continue;
    }
    int expected = p->user_cols[i].sql_type;
    bool ok = (vt == expected) ||
              (expected == SQLITE_FLOAT && vt == SQLITE_INTEGER);
    if (!ok) {
      *pzErr = sqlite3_mprintf(
          "column \"%s\" strict type mismatch: expected type %d, got %d",
          p->user_cols[i].name, expected, vt);
      return SQLITE_CONSTRAINT;
    }
    if (vt == SQLITE_TEXT) {
      const char *raw = (const char *)sqlite3_value_text(v);
      int rlen = sqlite3_value_bytes(v);
      if (!raw) {
        return SQLITE_NOMEM;
      }
      if ((size_t)rlen > SVE_MAX_TEXT_BYTES ||
          ((p->redact_text || p->user_cols[i].is_fulltext) &&
           memchr(raw, 0, (size_t)rlen) != NULL)) {
        *pzErr = sqlite3_mprintf("invalid or oversized text in column \"%s\"",
                                 p->user_cols[i].name);
        return SQLITE_CONSTRAINT;
      }
      if (p->redact_text && rlen > 0) {
        size_t cap = (size_t)rlen * 2U + 64U;
        char *buf = (char *)malloc(cap);
        if (!buf) {
          return SQLITE_NOMEM;
        }
        if (sve_privacy_scrub(raw, (size_t)rlen, buf, cap, NULL) != SVE_OK) {
          free(buf);
          return SQLITE_TOOBIG;
        }
        scrubbed_texts[i] = buf;
      }
    }
  }
  return SQLITE_OK;
}

static int delete_shadow_row(sve_vtab_t *p, int64_t rowid, bool fts) {
  char *sql = sqlite3_mprintf(
      "DELETE FROM \"%w\".\"%w_%s\" WHERE rowid = ?1;",
      p->schema_name, p->table_name, fts ? "fts" : "rows");
  if (!sql) {
    return SQLITE_NOMEM;
  }
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2(p->db, sql, -1, &stmt, NULL);
  sqlite3_free(sql);
  if (rc == SQLITE_OK) {
    rc = sqlite3_bind_int64(stmt, 1, rowid);
  }
  if (rc == SQLITE_OK) {
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_DONE) {
      rc = fts || sqlite3_changes64(p->db) == 1 ? SQLITE_OK : SQLITE_CORRUPT;
    }
  }
  sqlite3_finalize(stmt);
  return rc;
}

static int delete_shadow_rows(sve_vtab_t *p, int64_t rowid) {
  int rc = delete_shadow_row(p, rowid, false);
  if (rc == SQLITE_OK && p->fulltext_col_idx >= 0) {
    rc = delete_shadow_row(p, rowid, true);
  }
  return rc;
}

static int bind_user_value(sqlite3_stmt *stmt, int param, sqlite3_value *value,
                           const char *scrubbed) {
  return scrubbed ? sqlite3_bind_text(stmt, param, scrubbed, -1, SQLITE_STATIC)
                  : sqlite3_bind_value(stmt, param, value);
}

static int insert_shadow_row(sve_vtab_t *p, sqlite3_value **argv,
                             char *const *scrubbed, void *vec, size_t vec_bytes,
                             bool has_id, int64_t *id) {
  sqlite3_str *ins = sqlite3_str_new(p->db);
  sqlite3_str_appendf(ins, "INSERT INTO \"%w\".\"%w_rows\"(id, vec",
                      p->schema_name, p->table_name);
  for (size_t i = 0U; i < p->n_user_cols; i++) {
    sqlite3_str_appendf(ins, ", c%d", (int)i);
  }
  sqlite3_str_appendall(ins, ") VALUES (?, ?");
  for (size_t i = 0U; i < p->n_user_cols; i++) {
    sqlite3_str_appendall(ins, ", ?");
  }
  sqlite3_str_appendall(ins, ");");
  int rc = sqlite3_str_errcode(ins);
  char *sql = sqlite3_str_finish(ins);
  sqlite3_stmt *stmt = NULL;
  if (rc == SQLITE_OK) {
    rc = sqlite3_prepare_v2(p->db, sql, -1, &stmt, NULL);
  }
  sqlite3_free(sql);
  if (rc == SQLITE_OK && has_id) {
    rc = sqlite3_bind_int64(stmt, 1, *id);
  }
  if (rc == SQLITE_OK) {
    rc = sqlite3_bind_blob64(stmt, 2, vec, (sqlite3_uint64)vec_bytes, SQLITE_STATIC);
  }
  for (size_t i = 0U; rc == SQLITE_OK && i < p->n_user_cols; i++) {
    rc = bind_user_value(stmt, 3 + (int)i, argv[3U + i], scrubbed[i]);
  }
  if (rc == SQLITE_OK) {
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_DONE) {
      rc = sqlite3_changes64(p->db) == 1 ? SQLITE_OK : SQLITE_ERROR;
      if (rc == SQLITE_OK && !has_id) {
        *id = sqlite3_last_insert_rowid(p->db);
      }
    }
  }
  sqlite3_finalize(stmt);
  return rc;
}

static int insert_shadow_fts(sve_vtab_t *p, int64_t id, sqlite3_value *value,
                             const char *scrubbed) {
  if (sqlite3_value_type(value) == SQLITE_NULL) {
    return SQLITE_OK;
  }
  char *sql = sqlite3_mprintf(
      "INSERT INTO \"%w\".\"%w_fts\"(rowid, doc_text) VALUES (?1, ?2);",
      p->schema_name, p->table_name);
  if (!sql) {
    return SQLITE_NOMEM;
  }
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2(p->db, sql, -1, &stmt, NULL);
  sqlite3_free(sql);
  if (rc == SQLITE_OK) {
    rc = sqlite3_bind_int64(stmt, 1, id);
  }
  if (rc == SQLITE_OK) {
    rc = bind_user_value(stmt, 2, value, scrubbed);
  }
  if (rc == SQLITE_OK) {
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_DONE) {
      rc = sqlite3_changes64(p->db) == 1 ? SQLITE_OK : SQLITE_ERROR;
    }
  }
  sqlite3_finalize(stmt);
  return rc;
}

/* Only a constraint discovered before writes may obey IGNORE or FAIL. */
static int write_error(int rc) {
  return (rc & 0xFF) == SQLITE_CONSTRAINT ? SQLITE_ERROR : rc;
}

typedef struct shadow_write {
  sve_vtab_t *table;
  sqlite3_value **argv;
  char *const *scrubbed;
  void *vec;
  size_t vec_bytes;
  int64_t id;
  int64_t old_id;
  bool has_id;
  bool is_update;
  bool replace;
  bool deleting;
  bool applied;
} shadow_write_t;

static const char k_shadow_write_type[] = "sve0.shadow_write";

static void apply_shadow_write(sqlite3_context *ctx, int argc,
                               sqlite3_value **argv) {
  SVE_UNUSED(argc);
  shadow_write_t *write = (shadow_write_t *)sqlite3_value_pointer(
      argv[0], k_shadow_write_type);
  if (!write || write->applied ||
      sqlite3_context_db_handle(ctx) != write->table->db) {
    sqlite3_result_error(ctx, "invalid sve0 shadow operation", -1);
    return;
  }
  write->applied = true;
  sve_vtab_t *p = write->table;
  int rc = SQLITE_OK;
  if (write->is_update || write->deleting) {
    rc = delete_shadow_rows(p, write->old_id);
  }
  if (rc == SQLITE_OK && write->replace) {
    rc = delete_shadow_rows(p, write->id);
  }
  if (rc == SQLITE_OK && !write->deleting) {
    rc = insert_shadow_row(p, write->argv, write->scrubbed, write->vec,
                           write->vec_bytes, write->has_id, &write->id);
    if (rc == SQLITE_OK && p->fulltext_col_idx >= 0) {
      size_t col = (size_t)p->fulltext_col_idx;
      rc = insert_shadow_fts(p, write->id, write->argv[3U + col],
                             write->scrubbed[col]);
    }
  }
  if (rc == SQLITE_OK) {
    if (write->is_update || write->deleting) {
      rc = sqlite_index_error(sve_hnsw_delete(p->hnsw, write->old_id));
    }
    if (rc == SQLITE_OK && write->replace) {
      rc = sqlite_index_error(sve_hnsw_delete(p->hnsw, write->id));
    }
    if (rc == SQLITE_OK && !write->deleting) {
      rc = sqlite_index_error(sve_hnsw_insert(p->hnsw, write->id, write->vec));
    }
    p->hnsw_dirty = true;
    p->hnsw_rebuild = rc != SQLITE_OK;
  }
  if (rc != SQLITE_OK) {
    sqlite3_result_error_code(ctx, write_error(rc));
  } else {
    sqlite3_result_int(ctx, 0);
  }
}

static int run_shadow_write(shadow_write_t *write) {
  sve_vtab_t *p = write->table;
  /* INSERT SELECT with a uniqueness check gets a SQLite statement journal
     even when the outer single-row virtual-table statement has none. The
     callback returns false on success, so this guard inserts no record. */
  char *sql = sqlite3_mprintf(
      "INSERT OR ABORT INTO \"%w\".\"%w_info\"(key,val) "
      "SELECT 'hnsw_snapshot', NULL WHERE sve0_apply_shadow_write(?1);",
      p->schema_name, p->table_name);
  if (!sql) {
    return SQLITE_NOMEM;
  }
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2(p->db, sql, -1, &stmt, NULL);
  sqlite3_free(sql);
  if (rc == SQLITE_OK) {
    rc = sqlite3_bind_pointer(stmt, 1, write, k_shadow_write_type, NULL);
  }
  if (rc == SQLITE_OK) {
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_DONE) {
      rc = write->applied ? SQLITE_OK : SQLITE_ERROR;
    }
  }
  sqlite3_finalize(stmt);
  return write_error(rc);
}

static int sve_xUpdate(sqlite3_vtab *pVTab, int argc, sqlite3_value **argv,
                       sqlite_int64 *pRowid) {
  sve_vtab_t *p = (sve_vtab_t *)pVTab;
  int rc = refresh_hnsw(p, &pVTab->zErrMsg);
  if (rc != SQLITE_OK) {
    return rc;
  }

  if (argc == 1) {
    shadow_write_t write = {.table = p, .deleting = true,
                             .old_id = sqlite3_value_int64(argv[0])};
    return run_shadow_write(&write);
  }
  if (argc != 3 + (int)p->n_user_cols + 5) {
    return SQLITE_ERROR;
  }
  for (int col = p->col_distance; col <= p->col_rrf_score; col++) {
    if (sqlite3_value_type(argv[2 + col]) != SQLITE_NULL) {
      sqlite3_free(pVTab->zErrMsg);
      pVTab->zErrMsg = sqlite3_mprintf("cannot write sve0 hidden columns");
      return SQLITE_CONSTRAINT;
    }
  }

  bool is_update = sqlite3_value_type(argv[0]) != SQLITE_NULL;
  bool has_id = sqlite3_value_type(argv[1]) != SQLITE_NULL;
  if (has_id && sqlite3_value_numeric_type(argv[1]) != SQLITE_INTEGER) {
    sqlite3_free(pVTab->zErrMsg);
    pVTab->zErrMsg = sqlite3_mprintf("sve0 rowid must be an integer");
    return SQLITE_CONSTRAINT;
  }
  int64_t id = has_id ? sqlite3_value_int64(argv[1]) : 0;
  int64_t old_id = is_update ? sqlite3_value_int64(argv[0]) : 0;
  bool replace = has_id && (!is_update || old_id != id) &&
                 sve_hnsw_contains(p->hnsw, id);
  if (replace && sqlite3_vtab_on_conflict(p->db) != SQLITE_REPLACE) {
    sqlite3_free(pVTab->zErrMsg);
    pVTab->zErrMsg = sqlite3_mprintf("UNIQUE constraint failed on rowid %lld",
                                     (long long)id);
    return SQLITE_CONSTRAINT;
  }

  void *vec = NULL;
  size_t vec_bytes = 0U;
  char *err = NULL;
  char *scrubbed[SVE_MAX_USER_COLS] = {0};
  rc = sve_coerce_vector_value(argv[2], p->elem_type, p->dims, &vec, &vec_bytes,
                               &err);
  if (rc == SQLITE_OK) {
    rc = validate_strict_user_values(p, argc, argv, scrubbed, &err);
  }
  if (rc != SQLITE_OK) {
    sqlite3_free(pVTab->zErrMsg);
    pVTab->zErrMsg = err;
    goto done;
  }

  shadow_write_t write = {
      .table = p, .argv = argv, .scrubbed = scrubbed, .vec = vec,
      .vec_bytes = vec_bytes, .id = id, .old_id = old_id, .has_id = has_id,
      .is_update = is_update, .replace = replace,
  };
  rc = run_shadow_write(&write);
  if (rc == SQLITE_OK && pRowid) {
    *pRowid = write.id;
  }

done:
  for (size_t i = 0U; i < p->n_user_cols; i++) {
    free(scrubbed[i]);
  }
  free(vec);
  return rc;
}

static int sve_xSavepoint(sqlite3_vtab *pVTab, int iSavepoint) {
  SVE_UNUSED(pVTab);
  SVE_UNUSED(iSavepoint);
  return SQLITE_OK;
}

static int sve_xRelease(sqlite3_vtab *pVTab, int iSavepoint) {
  SVE_UNUSED(pVTab);
  SVE_UNUSED(iSavepoint);
  return SQLITE_OK;
}

static int sve_xRollbackTo(sqlite3_vtab *pVTab, int iSavepoint) {
  SVE_UNUSED(iSavepoint);
  sve_vtab_t *p = (sve_vtab_t *)pVTab;
  p->hnsw_rebuild = true;
  p->hnsw_dirty = true;
  return SQLITE_OK;
}

static int sve_xShadowName(const char *zName) {
  static const char *const k_shadows[] = {"info", "rows", "fts"};
  for (size_t i = 0U; i < sizeof(k_shadows) / sizeof(k_shadows[0]); i++) {
    if (sqlite3_stricmp(zName, k_shadows[i]) == 0) {
      return 1;
    }
  }
  return 0;
}

static sqlite3_module g_sve0_module = {
    3,                /* iVersion */
    sve_xCreate,      /* xCreate */
    sve_xConnect,     /* xConnect */
    sve_xBestIndex,   /* xBestIndex */
    sve_xDisconnect,  /* xDisconnect */
    sve_xDestroy,     /* xDestroy */
    sve_xOpen,        /* xOpen */
    sve_xClose,       /* xClose */
    sve_xFilter,      /* xFilter */
    sve_xNext,        /* xNext */
    sve_xEof,         /* xEof */
    sve_xColumn,      /* xColumn */
    sve_xRowid,       /* xRowid */
    sve_xUpdate,      /* xUpdate */
    sve_xBegin,       /* xBegin */
    sve_xSync,        /* xSync */
    sve_xCommit,      /* xCommit */
    sve_xRollback,    /* xRollback */
    sve_xFindFunction, /* xFindFunction */
    sve_xRename,     /* xRename */
    sve_xSavepoint,   /* xSavepoint */
    sve_xRelease,     /* xRelease */
    sve_xRollbackTo,  /* xRollbackTo */
    sve_xShadowName,  /* xShadowName */
    NULL              /* xIntegrity */
};

int sve_register_vtab_module(sqlite3 *db, char **pzErrMsg) {
  if (!db) {
    return SQLITE_MISUSE;
  }
  int rc = sqlite3_create_function_v2(
      db, "sve0_apply_shadow_write", 1, SQLITE_UTF8 | SQLITE_DIRECTONLY, NULL,
      apply_shadow_write, NULL, NULL, NULL);
  if (rc == SQLITE_OK) {
    rc = sqlite3_create_module_v2(db, "sve0", &g_sve0_module, NULL, NULL);
    if (rc != SQLITE_OK) {
      sqlite3_create_function_v2(db, "sve0_apply_shadow_write", 1, SQLITE_UTF8,
                                NULL, NULL, NULL, NULL, NULL);
    }
  }
  if (rc != SQLITE_OK && pzErrMsg) {
    *pzErrMsg = sqlite3_mprintf("failed to create sve0 module: %s",
                                sqlite3_errmsg(db));
  }
  return rc;
}
