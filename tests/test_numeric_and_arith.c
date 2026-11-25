#include "test_common.h"
#include "sqlite3.h"
#include "sve_hybrid/sve_numeric.h"
#include "sve_hybrid/sve_sqlite_ext.h"

#include <limits.h>
#include <stdint.h>

static void test_l2_scale(void) {
  sqlite3 *db = NULL;
  SVE_ASSERT_EQ_INT(sqlite3_open(":memory:", &db), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sve_hybrid_register(db, NULL), SQLITE_OK);

  sqlite3_stmt *st = NULL;
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db, "SELECT sve_distance_l2('[1e20, 0, 0, 0]', '[0, 0, 0, 0]');",
          -1, &st, NULL),
      SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_ROW);
  double d_large = sqlite3_column_double(st, 0);
  SVE_ASSERT(!isinf(d_large) && !isnan(d_large));
  SVE_ASSERT_NEAR(d_large, 1e20, 1e14);
  sqlite3_finalize(st);

  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db, "SELECT sve_distance_l2('[1e-25, 0, 0, 0]', '[0, 0, 0, 0]');",
          -1, &st, NULL),
      SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_ROW);
  double d_tiny = sqlite3_column_double(st, 0);
  SVE_ASSERT(d_tiny > 0.0);
  SVE_ASSERT_NEAR(d_tiny, 1e-25, 1e-30);
  sqlite3_finalize(st);

  /* NaN rejection via SQL JSON parser */
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db, "SELECT sve_distance_l2('[NaN, 0]', '[0, 0]');", -1, &st, NULL),
      SQLITE_OK);
  SVE_ASSERT(sqlite3_step(st) != SQLITE_ROW);
  sqlite3_finalize(st);

  sqlite3_close(db);
}

static void test_l1_int64(void) {
  /* Direct C kernel + SQL int64 path > INT32_MAX */
  const size_t dims = 8500000U;
  int8_t *a = (int8_t *)malloc(dims);
  int8_t *b = (int8_t *)malloc(dims);
  SVE_ASSERT(a != NULL && b != NULL);
  memset(a, 127, dims);
  memset(b, (uint8_t)(int8_t)(-128), dims);

  int64_t l1_c = 0;
  SVE_ASSERT_EQ_INT(sve_distance_l1_i8(a, b, dims, &l1_c), SVE_OK);
  SVE_ASSERT_EQ_INT(l1_c, (int64_t)dims * 255LL);

  sqlite3 *db = NULL;
  SVE_ASSERT_EQ_INT(sqlite3_open(":memory:", &db), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sve_hybrid_register(db, NULL), SQLITE_OK);

  sqlite3_stmt *st = NULL;
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db, "SELECT sve_distance_l1(sve_int8(?1), sve_int8(?2));", -1, &st,
          NULL),
      SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_bind_blob(st, 1, a, (int)dims, SQLITE_STATIC),
                    SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_bind_blob(st, 2, b, (int)dims, SQLITE_STATIC),
                    SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_ROW);
  sqlite3_int64 l1_sql = sqlite3_column_int64(st, 0);
  SVE_ASSERT_EQ_INT(l1_sql, 2167500000LL);
  sqlite3_finalize(st);

  sqlite3_close(db);
  free(a);
  free(b);
}

static void test_cosine_zero(void) {
  sqlite3 *db = NULL;
  SVE_ASSERT_EQ_INT(sqlite3_open(":memory:", &db), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sve_hybrid_register(db, NULL), SQLITE_OK);

  sqlite3_stmt *st = NULL;
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db, "SELECT sve_distance_cosine('[0, 0, 0]', '[1, 2, 3]');", -1, &st,
          NULL),
      SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_ROW);
  double c = sqlite3_column_double(st, 0);
  SVE_ASSERT(!isnan(c));
  SVE_ASSERT_NEAR(c, 1.0, 1e-12);
  sqlite3_finalize(st);

  /* Identical unit vectors stay bounded >= 0.0 (never negative fp error) */
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db,
          "SELECT sve_distance_cosine('[0.57735027, 0.57735027, 0.57735027]', "
          "'[0.57735027, 0.57735027, 0.57735027]');",
          -1, &st, NULL),
      SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_ROW);
  double c2 = sqlite3_column_double(st, 0);
  SVE_ASSERT(c2 >= 0.0 && c2 < 1e-6);
  sqlite3_finalize(st);
  sqlite3_close(db);
}

static void test_arith_checked(void) {
  sqlite3 *db = NULL;
  SVE_ASSERT_EQ_INT(sqlite3_open(":memory:", &db), SQLITE_OK);
  SVE_ASSERT_EQ_INT(sve_hybrid_register(db, NULL), SQLITE_OK);

  /* Checked int8 add rejects 100 + 50 > 127 */
  sqlite3_stmt *st = NULL;
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db, "SELECT sve_add(sve_int8('[100, -100]'), sve_int8('[50, -50]'));",
          -1, &st, NULL),
      SQLITE_OK);
  SVE_ASSERT(sqlite3_step(st) != SQLITE_ROW);
  sqlite3_finalize(st);

  /* Saturating add clamps cleanly to [127, -128] */
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db,
          "SELECT sve_to_json(sve_add_sat(sve_int8('[100, -100]'), "
          "sve_int8('[50, -50]')));",
          -1, &st, NULL),
      SQLITE_OK);
  SVE_ASSERT_EQ_INT(sqlite3_step(st), SQLITE_ROW);
  SVE_ASSERT(strcmp((const char *)sqlite3_column_text(st, 0), "[127,-128]") == 0);
  sqlite3_finalize(st);

  /* Checked float32 add rejects overflow to Inf */
  SVE_ASSERT_EQ_INT(
      sqlite3_prepare_v2(
          db, "SELECT sve_add('[3e38, 0]', '[3e38, 0]');", -1, &st, NULL),
      SQLITE_OK);
  SVE_ASSERT(sqlite3_step(st) != SQLITE_ROW);
  sqlite3_finalize(st);

  sqlite3_close(db);
}

static void test_crc_sq8(void) {
  /* RFC 3720 / Castagnoli official test vector */
  uint32_t crc = sve_crc32c("123456789", 9U);
  SVE_ASSERT_EQ_INT((long long)crc, (long long)0xE3069283U);

  float src[4] = {1.0f, -0.5f, 0.25f, 0.0f};
  int8_t q[4] = {0};
  float scale = 0.0f;
  SVE_ASSERT_EQ_INT(sve_quantize_f32_to_sq8(src, 4U, q, &scale), SVE_OK);
  SVE_ASSERT_EQ_INT(q[0], 127);
  float rec[4] = {0};
  SVE_ASSERT_EQ_INT(sve_dequantize_sq8_to_f32(q, scale, 4U, rec), SVE_OK);
  for (size_t i = 0U; i < 4U; i++) {
    SVE_ASSERT_NEAR(rec[i], src[i], 0.02);
  }
}

static void test_json_limits(void) {
  float *fv = NULL;
  size_t dims = 0U;
  SVE_ASSERT(sve_parse_f32_json("[1.0, 2.0] trailing", 19U, 16U, &fv, &dims) !=
             SVE_OK);
  SVE_ASSERT(sve_parse_f32_json("[]", 2U, 16U, &fv, &dims) != SVE_OK);
  SVE_ASSERT(sve_parse_f32_json("[1,2,3,4,5]", 11U, 4U, &fv, &dims) ==
             SVE_ERR_LIMIT);
  SVE_ASSERT(sve_parse_f32_json("[1e500, 0]", 10U, 4U, &fv, &dims) != SVE_OK);

  int8_t *iv = NULL;
  SVE_ASSERT(sve_parse_i8_json("[128, 0]", 8U, 4U, &iv, &dims) != SVE_OK);
  SVE_ASSERT(sve_parse_i8_json("[-129, 0]", 9U, 4U, &iv, &dims) != SVE_OK);
  SVE_ASSERT_EQ_INT(sve_parse_i8_json("[-128, 127]", 11U, 4U, &iv, &dims),
                    SVE_OK);
  SVE_ASSERT_EQ_INT(dims, 2U);
  SVE_ASSERT_EQ_INT(iv[0], -128);
  SVE_ASSERT_EQ_INT(iv[1], 127);
  free(iv);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    test_l2_scale();
    test_l1_int64();
    test_cosine_zero();
    test_arith_checked();
    test_crc_sq8();
    test_json_limits();
    return 0;
  }
  if (strcmp(argv[1], "l2_scale") == 0) {
    test_l2_scale();
  } else if (strcmp(argv[1], "l1_int64") == 0) {
    test_l1_int64();
  } else if (strcmp(argv[1], "cosine_zero") == 0) {
    test_cosine_zero();
  } else if (strcmp(argv[1], "arith_checked") == 0) {
    test_arith_checked();
  } else if (strcmp(argv[1], "crc_sq8") == 0) {
    test_crc_sq8();
  } else if (strcmp(argv[1], "json_limits") == 0) {
    test_json_limits();
  } else {
    fprintf(stderr, "Unknown subtest: %s\n", argv[1]);
    return 1;
  }
  return 0;
}
