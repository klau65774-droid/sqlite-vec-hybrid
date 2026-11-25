# Topic Selection & Upgrade Proposal (`sqlite-vec-hybrid`)

## 1. Selected Indie Open-Source Baseline

- **Upstream Repository**: `asg017/sqlite-vec` (<https://github.com/asg017/sqlite-vec>)
- **Upstream Commit Pinned**: `04d28bd21773981e2d266bbf6aa4efbd011eb4f6` (`v0.1.10-alpha.4`)
- **Author & Governance**: Personal independent maintainer (`asg017`, `User`), dual-licensed under `Apache-2.0` / `MIT`. Excluded all big-tech/hyperscaler organizations.
- **Form Factor**: Mode A — Pure C11 core library, SQLite virtual-table extension (`sve0` + `sve_*`), CTest verification harness, and CLI benchmark runner (`sqlite3.c` 3.45.3 amalgamation with FTS5). Zero frontend/backend web split, zero paid LLM APIs, 100% local CPU execution.

## 2. Verified Reproducible Upstream Defects & Bottlenecks

All five defect classes are directly observable in `upstream/sqlite-vec/sqlite-vec.c`:

1. **Ghost Row Leak on Failed `vec0` Insert (`#333`, `sqlite-vec.c:9327-9452`)**:
   - `vec0Update_Insert()` writes `_rowids` and sets chunk validity bits + vector blobs BEFORE validating auxiliary column constraints (`line 9415`) and metadata columns (`line 9448`).
   - When an insert fails inside an explicit transaction (and the caller continues/commits other rows), a partial row remains searchable via KNN `MATCH`.
   - Also leaks `zSql` in auxiliary insert (`line 9396`, missing `sqlite3_free(zSql)`).
2. **Untyped `BLOB` Parameter Rejected by `int8` `vec0` Columns (`#361`, `sqlite-vec.c:1263-1275, 9246-9254`)**:
   - SQLite strips value subtypes across bound `BLOB` parameters, subqueries, and materialized columns; `vector_from_value()` falls back to `SQLITE_VEC_ELEMENT_TYPE_FLOAT32` whenever `subtype == 0`.
   - Valid `int8` blobs (including `vec_int8()` bytes passed back as bound BLOBs) fail with `"expected to be of type int8, but a float32 vector was provided"`, while `line 9252` misindexes `p->vector_columns[i]` with user-column index `i`.
3. **Float32 `L2` & `Cosine` Overflow/Underflow & Zero-Norm `NaN` (`#334`, `sqlite-vec.c:382-396, 484-523`)**:
   - `l2_sqr_float()` accumulates `t * t` in 32-bit `float`: magnitudes `~1e20` overflow `FLT_MAX` to `Infinity`; magnitudes `~1e-25` underflow subnormals to `0.0`.
   - `cosine_float()` uses `float32` accumulators and divides `0 / 0 -> NaN` on zero vectors.
4. **`int8` `L1` `INT32_MAX` Overflow & Result Truncation (`#362`, `sqlite-vec.c:436-449, 1534-1538`)**:
   - `l1_int8()` accumulates into signed `i32 res` (signed overflow UB / wraps at `INT32_MAX`) and `vec_distance_l1()` passes it to `sqlite3_result_int()` instead of `sqlite3_result_int64()`.
5. **Unchecked `vec_add` Wrap / Non-Finite Propagation & Brute-Force Scaling (`#363`, `sqlite-vec.c:1756-1785`)**:
   - `vec_add()` wraps `int8` modulo 256 (signed overflow UB) and emits `Inf`/`NaN` `float32` vectors.
   - Only $O(N)$ linear scan exists for stable general use, post-filtering suffers recall collapse under selective metadata predicates, and there is no native FTS5 BM25 + dense ANN RRF hybrid ranking.

## 3. Target External Blind Benchmark Plan (Local CPU Only)

- **Tier 1 (Dense + Hybrid Retrieval Quality & Latency)**:
  - **BEIR `SciFact` & `NFCorpus`** (open scientific & biomedical IR corpora, Apache-2.0 / public research release): full BM25 (FTS5) + Dense Embeddings (`HNSW` vs `FLAT`) + Native `RRF` (`k=60`) hybrid fusion measuring **NDCG@10, MRR@10, Recall@10, and P50/P99 latency**.
  - **10K–50K Vector ANN Scaling Benchmark** (derived outside repo on local CPU, deterministic projection + quantized int8 / float32 HNSW): comparing `upstream vec0 O(N)` vs `sve0 HNSW` vs filtered `WHERE` selectivity (`5%` predicate match rate).
- **Tier 2 (Correctness, Numerical Stability & Persistence Integrity)**:
  - Upstream defect matrix (`Ghost Row`, `int8 bound BLOB`, `L2 1e20 / 1e-25`, `L1 INT32_MAX`, `vec_add saturation/rejection`) comparing unmodified `vec0` (`FAILED`) vs `sve0` (`PASSED`).
  - **HNSW Save/Reload CRC32C Shadow-Table Integrity & Corruption Detection** under simulated torn writes.
- **Tier 3 (Privacy Redaction & DoS Boundaries)**:
  - OOD 15-secret / 15-benign-decoy PII scrubber audit (`100%` recall, `0%` false positives) + dimension/JSON/HNSW parameter DoS guards.
