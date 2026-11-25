cmake_minimum_required(VERSION 3.20)
project(sqlite_vec_hybrid VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/utf-8)
endif()

# Official Public Domain SQLite 3.45.3 amalgamation with FTS5
add_library(sqlite3_core STATIC
  third_party/sqlite3/sqlite3.c
)
target_include_directories(sqlite3_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite3
)
target_compile_definitions(sqlite3_core PUBLIC
  SQLITE_ENABLE_FTS5
  SQLITE_ENABLE_RTREE
)
if(NOT MSVC)
  target_compile_options(sqlite3_core PRIVATE -w)
else()
  target_compile_options(sqlite3_core PRIVATE /W0)
endif()

# 100% unmodified upstream baseline (asg017/sqlite-vec commit 04d28bd)
add_library(upstream_vec STATIC
  upstream/sqlite-vec/sqlite-vec.c
)
target_include_directories(upstream_vec PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/upstream/shim
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite3
)
target_compile_definitions(upstream_vec PUBLIC
  SQLITE_CORE
  SQLITE_VEC_STATIC
)
if(NOT MSVC)
  target_compile_options(upstream_vec PRIVATE -w)
else()
  target_compile_options(upstream_vec PRIVATE /W0)
endif()
target_link_libraries(upstream_vec PUBLIC sqlite3_core)

set(SVE_CORE_SOURCES
  src/sve_numeric.c
  src/sve_hnsw.c
  src/sve_hybrid_rank.c
  src/sve_vtab_guard.c
  src/sve_sqlite_ext.c
)

function(sve_enable_strict_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /WX)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
endfunction()

# Static core library linked against bundled sqlite3_core
add_library(sve_hybrid STATIC ${SVE_CORE_SOURCES})
target_include_directories(sve_hybrid PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite3
)
target_compile_definitions(sve_hybrid PUBLIC
  SQLITE_CORE
  SVE_HYBRID_STATIC
)
target_link_libraries(sve_hybrid PUBLIC sqlite3_core)
sve_enable_strict_warnings(sve_hybrid)
if(UNIX)
  target_link_libraries(sve_hybrid PUBLIC m)
endif()

# Standalone SQLite loadable shared extension (.dll / .so / .dylib)
add_library(sve_hybrid_mod SHARED ${SVE_CORE_SOURCES})
set_target_properties(sve_hybrid_mod PROPERTIES OUTPUT_NAME "vec_hybrid")
target_include_directories(sve_hybrid_mod PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite3
)
sve_enable_strict_warnings(sve_hybrid_mod)
if(UNIX)
  target_link_libraries(sve_hybrid_mod PRIVATE m)
endif()

# Unified benchmark / audit CLI
add_executable(sve_hybrid_cli cli/sve_hybrid_cli.c)
target_link_libraries(sve_hybrid_cli PRIVATE sve_hybrid upstream_vec sqlite3_core)
sve_enable_strict_warnings(sve_hybrid_cli)
if(UNIX)
  target_link_libraries(sve_hybrid_cli PRIVATE m)
endif()

enable_testing()

set(SVE_TEST_TARGETS
  test_upstream_repro
  test_numeric_and_arith
  test_vtab_atomicity
  test_hnsw_and_filter
  test_hybrid_and_privacy
  test_audit_regression
  test_hnsw_alloc
)

foreach(t IN LISTS SVE_TEST_TARGETS)
  add_executable(${t} tests/${t}.c)
  target_include_directories(${t} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/upstream/shim
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite3
  )
  target_link_libraries(${t} PRIVATE sve_hybrid upstream_vec sqlite3_core)
  sve_enable_strict_warnings(${t})
  if(UNIX)
    target_link_libraries(${t} PRIVATE m)
  endif()
endforeach()

add_test(NAME upstream_repro COMMAND test_upstream_repro)

add_test(NAME numeric_l2_scale COMMAND test_numeric_and_arith l2_scale)
add_test(NAME numeric_l1_int64 COMMAND test_numeric_and_arith l1_int64)
add_test(NAME numeric_cosine_zero COMMAND test_numeric_and_arith cosine_zero)
add_test(NAME numeric_arith_checked COMMAND test_numeric_and_arith arith_checked)
add_test(NAME numeric_crc32c_and_sq8 COMMAND test_numeric_and_arith crc_sq8)
add_test(NAME numeric_json_limits COMMAND test_numeric_and_arith json_limits)

add_test(NAME vtab_no_ghost_row COMMAND test_vtab_atomicity ghost_row)
add_test(NAME vtab_bound_int8_blob COMMAND test_vtab_atomicity bound_int8)
add_test(NAME vtab_replace_update_delete COMMAND test_vtab_atomicity crud)
add_test(NAME vtab_rollback_safety COMMAND test_vtab_atomicity rollback)
add_test(NAME vtab_disk_persistence COMMAND test_vtab_atomicity disk_persist)

add_test(NAME hnsw_recall_1k COMMAND test_hnsw_and_filter recall_1k)
add_test(NAME hnsw_sq8_quantized COMMAND test_hnsw_and_filter sq8)
add_test(NAME hnsw_selective_prefilter COMMAND test_hnsw_and_filter filter)
add_test(NAME hnsw_snapshot_crc_corrupt COMMAND test_hnsw_and_filter crc_corrupt)

add_test(NAME hybrid_rrf_and_convex COMMAND test_hybrid_and_privacy fusion)
add_test(NAME hybrid_fts5_sanitizer COMMAND test_hybrid_and_privacy fts_sanitize)
add_test(NAME privacy_ood_15_15 COMMAND test_hybrid_and_privacy privacy)
add_test(NAME ext_loadable_dll COMMAND test_hybrid_and_privacy $<TARGET_FILE:sve_hybrid_mod>)

add_test(
  NAME cli_audit_roundtrip
  COMMAND ${CMAKE_COMMAND}
    -DCLI=$<TARGET_FILE:sve_hybrid_cli>
    -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cli_audit.cmake
)

add_test(NAME audit_json_locale_and_numeric_extremes COMMAND test_audit_regression json)
add_test(NAME audit_rank_and_fts_utf8 COMMAND test_audit_regression rank)
add_test(NAME audit_privacy_scan_boundaries COMMAND test_audit_regression privacy)
add_test(NAME audit_vtab_metadata_plan COMMAND test_audit_regression planner)
add_test(NAME audit_vtab_query_controls COMMAND test_audit_regression controls)
add_test(NAME audit_vtab_shadow_write_failure COMMAND test_audit_regression shadow_failure)
add_test(NAME audit_vtab_savepoint_conflicts COMMAND test_audit_regression savepoint)
add_test(NAME audit_vtab_connection_visibility COMMAND test_audit_regression connections)
add_test(NAME audit_vtab_shadow_ownership_identifiers COMMAND test_audit_regression identifiers)
add_test(NAME audit_vtab_snapshot_connect COMMAND test_audit_regression snapshot)
add_test(NAME audit_abi_static_subtypes COMMAND test_audit_regression subtypes)
add_test(NAME audit_abi_loadable_subtypes
  COMMAND test_audit_regression subtypes $<TARGET_FILE:sve_hybrid_mod>)
add_test(NAME audit_hnsw_insert_oom COMMAND test_hnsw_alloc insert)
add_test(NAME audit_hnsw_legacy_snapshot COMMAND test_hnsw_alloc legacy)
add_test(NAME audit_hnsw_entry_allowlist COMMAND test_hnsw_alloc entry)
add_test(NAME audit_vtab_hnsw_insert_failure COMMAND test_hnsw_alloc vtab)
add_test(
  NAME audit_cli_input_validation
  COMMAND ${CMAKE_COMMAND}
    -DCLI=$<TARGET_FILE:sve_hybrid_cli>
    -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cli_boundaries.cmake
)
