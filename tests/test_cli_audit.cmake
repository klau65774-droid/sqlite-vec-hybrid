if(NOT CLI)
  message(FATAL_ERROR "CLI path required")
endif()

execute_process(
  COMMAND "${CLI}" version
  RESULT_VARIABLE rc_ver
  OUTPUT_VARIABLE out_ver
)
if(NOT rc_ver EQUAL 0 OR NOT out_ver MATCHES "sve-hybrid")
  message(FATAL_ERROR "version failed: ${rc_ver} ${out_ver}")
endif()

execute_process(
  COMMAND "${CLI}" repro-compare
  RESULT_VARIABLE rc_repro
  OUTPUT_VARIABLE out_repro
)
if(NOT rc_repro EQUAL 0 OR NOT out_repro MATCHES "\"hybrid_ghost_count\": 0" OR NOT out_repro MATCHES "\"hybrid_accepted\": true")
  message(FATAL_ERROR "repro-compare failed: ${rc_repro} ${out_repro}")
endif()

execute_process(
  COMMAND "${CLI}" privacy-audit
  RESULT_VARIABLE rc_priv
  OUTPUT_VARIABLE out_priv
)
if(NOT rc_priv EQUAL 0 OR NOT out_priv MATCHES "\"secret_recall\":\"15/15\".*\"benign_fp\":\"0/15\"")
  message(FATAL_ERROR "privacy-audit failed: ${rc_priv} ${out_priv}")
endif()

execute_process(
  COMMAND "${CLI}" ann-bench --n 400 --dims 32 --k 10 --queries 20 --filter-pct 10 --seed 7
  RESULT_VARIABLE rc_bench
  OUTPUT_VARIABLE out_bench
)
if(NOT rc_bench EQUAL 0 OR NOT out_bench MATCHES "\"recall_at_k\"")
  message(FATAL_ERROR "ann-bench failed: ${rc_bench} ${out_bench}")
endif()
