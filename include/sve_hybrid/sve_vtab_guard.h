#ifndef SVE_VTAB_GUARD_H
#define SVE_VTAB_GUARD_H

#include "sve_config.h"
#ifndef SQLITE_CORE
#include "sqlite3ext.h"
#else
#include "sqlite3.h"
#endif
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Outbound / ingestion privacy scrubber:
 * Redacts PEM private keys, Bearer/JWT tokens, API keys, DSN passwords,
 * labeled IMSI/IMEI/MSISDN identifiers, and formatted phone literals while
 * keeping benign UUID4, SHA-256 hashes, ISO dates, and order IDs untouched.
 */
int sve_privacy_scrub(const char *input, size_t input_len, char *out_buf,
                      size_t out_cap, size_t *out_redactions);

/*
 * Coerces an SQLite value into vector bytes of the expected element type & dims.
 * Supports subtyped values (sve_f32/vec_f32/sve_int8/vec_int8/vec_bit),
 * JSON arrays, AND untyped bound BLOBs whose byte length matches expected_type
 * (#361 bound int8 BLOB fix).
 */
int sve_coerce_vector_value(sqlite3_value *val, sve_elem_type_t expected_type,
                            size_t expected_dims, void **out_buf,
                            size_t *out_bytes, char **pzErr);

/* Registers the `sve0` hardened hybrid virtual table module on `db`. */
int sve_register_vtab_module(sqlite3 *db, char **pzErrMsg);

#ifdef __cplusplus
}
#endif

#endif /* SVE_VTAB_GUARD_H */
