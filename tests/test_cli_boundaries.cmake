if(NOT CLI)
  message(FATAL_ERROR "CLI path required")
endif()

execute_process(
  COMMAND "${CLI}" ann-bench --n -1
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE out
)
if(rc EQUAL 0 OR NOT out STREQUAL "")
  message(FATAL_ERROR "Invalid ANN argument accepted")
endif()

set(tsv "audit_cli_eval.tsv")
file(WRITE "${tsv}" "DOC\t1\t0000803f00000000\tsample\nQRY\t1\t0000803f00000000\tsample\t1\n")
execute_process(
  COMMAND "${CLI}" hybrid-eval --input "${tsv}" --dims 2
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE out
)
if(NOT rc EQUAL 0 OR NOT out MATCHES "\"docs\": 1, \"queries\": 1")
  message(FATAL_ERROR "Valid benchmark records rejected")
endif()

file(WRITE "${tsv}" "DOC\t1\t0000803f00000000\tsample\nQRY\t1\t0000803f00000000\tsample\tinvalid\n")
execute_process(
  COMMAND "${CLI}" hybrid-eval --input "${tsv}" --dims 2
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE out
  ERROR_VARIABLE err
)
file(REMOVE "${tsv}")
if(rc EQUAL 0 OR NOT out STREQUAL "" OR NOT err MATCHES "benchmark record at line 2")
  message(FATAL_ERROR "Invalid benchmark record did not fail closed")
endif()
