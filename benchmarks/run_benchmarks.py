#!/usr/bin/env python3
"""
Three-tier local benchmark runner for sqlite-vec-hybrid.

Runs 100% locally on CPU with zero paid APIs:
  Tier 1: BEIR SciFact & NFCorpus (FTS5 BM25 vs upstream vec0 vs sve0 HNSW vs sve0 Hybrid RRF)
  Tier 2: 20,000-vector ANN + 5% filtered KNN scaling + upstream defect comparison
  Tier 3: OOD Privacy Scrubbing (15/15 secret recall, 0/15 benign false positives)
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")


def build_tfidf_svd_embeddings(
    corpus_texts: List[str],
    query_texts: List[str],
    dims: int = 128,
    max_vocab: int = 24000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit corpus-only TF-IDF + TruncatedSVD(dims) on CPU with zero query leakage."""
    doc_tokens: List[List[str]] = []
    df_counts: collections.Counter[str] = collections.Counter()
    for txt in corpus_texts:
        toks = TOKEN_RE.findall(txt.lower())
        doc_tokens.append(toks)
        for t in set(toks):
            df_counts[t] += 1

    n_docs = len(corpus_texts)
    min_df = 2
    max_df = max(min_df + 1, int(0.85 * n_docs))
    vocab_items = [
        (tok, df)
        for tok, df in df_counts.items()
        if min_df <= df <= max_df
    ]
    vocab_items.sort(key=lambda kv: (-kv[1], kv[0]))
    vocab_items = vocab_items[:max_vocab]
    vocab: Dict[str, int] = {tok: i for i, (tok, _) in enumerate(vocab_items)}
    n_vocab = len(vocab)
    idf = np.zeros(n_vocab, dtype=np.float64)
    for tok, idx in vocab.items():
        df = df_counts[tok]
        idf[idx] = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0

    def to_tfidf(token_lists: List[List[str]]) -> sp.csr_matrix:
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        for r, toks in enumerate(token_lists):
            tf = collections.Counter(t for t in toks if t in vocab)
            if not tf:
                continue
            for tok, cnt in tf.items():
                rows.append(r)
                cols.append(vocab[tok])
                vals.append((1.0 + math.log(cnt)) * idf[vocab[tok]])
        mat = sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(len(token_lists), n_vocab),
            dtype=np.float64,
        )
        # Row L2 normalize sparse TF-IDF vectors
        sq = mat.multiply(mat).sum(axis=1).A1
        inv = np.where(sq > 0.0, 1.0 / np.sqrt(np.maximum(sq, 1e-12)), 0.0)
        return sp.diags(inv) @ mat

    x_corpus = to_tfidf(doc_tokens)
    q_tokens = [TOKEN_RE.findall(txt.lower()) for txt in query_texts]
    x_queries = to_tfidf(q_tokens)

    k = min(dims, min(x_corpus.shape) - 1)
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(min(x_corpus.shape))
    u, s, vt = spla.svds(x_corpus, k=k, v0=v0, tol=1e-4)
    order = np.argsort(-s)
    s = s[order]
    u = u[:, order]
    vt = vt[order, :]

    doc_dense = u * s[np.newaxis, :]
    qry_dense = x_queries @ vt.T

    if k < dims:
        doc_dense = np.pad(doc_dense, ((0, 0), (0, dims - k)))
        qry_dense = np.pad(qry_dense, ((0, 0), (0, dims - k)))

    def l2_norm_f32(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        out = (arr / norms).astype("<f4")
        return np.ascontiguousarray(out)

    return l2_norm_f32(doc_dense), l2_norm_f32(qry_dense)


def prepare_beir_tsv(dataset_dir: Path, work_dir: Path, name: str, dims: int) -> Tuple[Path, int, int, dict[str, str]]:
    ds_path = dataset_dir / name
    corpus_file = ds_path / "corpus.jsonl"
    queries_file = ds_path / "queries.jsonl"
    qrels_file = ds_path / "qrels" / "test.tsv"
    if not corpus_file.exists() or not queries_file.exists() or not qrels_file.exists():
        raise FileNotFoundError(f"Missing BEIR dataset files for {name}")
    input_sha256 = {}
    for path in (corpus_file, queries_file, qrels_file):
        with path.open("rb") as source:
            input_sha256[path.relative_to(ds_path).as_posix()] = hashlib.file_digest(
                source, "sha256"
            ).hexdigest()

    doc_id_to_rowid: Dict[str, int] = {}
    corpus_texts: List[str] = []
    with corpus_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            did = str(obj["_id"])
            if did in doc_id_to_rowid:
                continue
            rowid = len(corpus_texts) + 1
            doc_id_to_rowid[did] = rowid
            raw = f"{obj.get('title', '')} {obj.get('text', '')}"
            clean = " ".join(raw.replace("\x00", " ").split())
            corpus_texts.append(clean)

    qrels: Dict[str, List[int]] = collections.defaultdict(list)
    with qrels_file.open("r", encoding="utf-8") as fh:
        header = fh.readline()
        _ = header
        for line in fh:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3:
                continue
            qid, did, score_s = parts[0], parts[1], parts[2]
            try:
                score = float(score_s)
            except ValueError:
                continue
            if score > 0.0 and did in doc_id_to_rowid:
                qrels[qid].append(doc_id_to_rowid[did])

    eval_queries: List[Tuple[str, str, List[int]]] = []
    with queries_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            qid = str(obj["_id"])
            if qid not in qrels:
                continue
            qtxt = " ".join(str(obj.get("text", "")).replace("\x00", " ").split())
            if qtxt:
                eval_queries.append((qid, qtxt, sorted(set(qrels[qid]))))

    q_texts = [q[1] for q in eval_queries]
    doc_vecs, qry_vecs = build_tfidf_svd_embeddings(corpus_texts, q_texts, dims=dims)

    out_tsv = work_dir / f"{name}_eval.tsv"
    with out_tsv.open("w", encoding="utf-8") as out:
        for i, txt in enumerate(corpus_texts):
            hex_v = doc_vecs[i].tobytes().hex()
            out.write(f"DOC\t{i + 1}\t{hex_v}\t{txt}\n")
        for idx, (_qid, qtxt, pos_ids) in enumerate(eval_queries):
            hex_q = qry_vecs[idx].tobytes().hex()
            pos_str = ",".join(str(x) for x in pos_ids)
            out.write(f"QRY\t{idx + 1}\t{hex_q}\t{qtxt}\t{pos_str}\n")

    return out_tsv, len(corpus_texts), len(eval_queries), input_sha256


def run_cli_json(cli: Path, args: List[str]) -> Dict[str, object]:
    proc = subprocess.run(
        [str(cli), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode and not (args[0] == "privacy-audit" and proc.stdout):
        raise RuntimeError(f"{args[0]} exited with status {proc.returncode}")
    result = json.loads(proc.stdout)
    if not isinstance(result, dict):
        raise ValueError(f"{args[0]} must return a JSON object")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sqlite-vec-hybrid 3-tier benchmarks")
    parser.add_argument(
        "--cli",
        default="build/sve_hybrid_cli.exe" if os.name == "nt" else "build/sve_hybrid_cli",
        help="Path to compiled sve_hybrid_cli binary",
    )
    parser.add_argument(
        "--dataset-dir",
        default=os.environ.get("SVE_BENCHMARK_DIR", ""),
        help="Outside-repo directory containing external BEIR datasets",
    )
    parser.add_argument(
        "--output",
        default="docs/benchmarks_report.json",
        help="Output JSON report path relative to repository root",
    )
    parser.add_argument("--dims", type=int, default=128)
    args = parser.parse_args()

    cli_path = Path(args.cli).resolve()
    if not cli_path.is_file():
        print("CLI executable not found", file=sys.stderr)
        return 1

    ds_dir = Path(args.dataset_dir) if args.dataset_dir else None
    if ds_dir is None or not ds_dir.exists():
        print("Pass --dataset-dir pointing to external BEIR datasets", file=sys.stderr)
        return 1

    if not 1 <= args.dims <= 8192:
        parser.error("--dims must be in [1, 8192]")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tier1: Dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix=".sve-bench-", dir=out_path.parent) as tmp:
        for ds_name in ("scifact", "nfcorpus"):
            tsv_path, n_docs, n_q, sha256 = prepare_beir_tsv(
                ds_dir, Path(tmp), ds_name, dims=args.dims
            )
            res = run_cli_json(
                cli_path,
                ["hybrid-eval", "--input", str(tsv_path), "--dims", str(args.dims)],
            )
            if (res["docs"], res["queries"], res["dims"]) != (n_docs, n_q, args.dims):
                raise ValueError(f"{ds_name} CLI counts do not match the prepared inputs")
            res["corpus_docs"] = n_docs
            res["eval_queries"] = n_q
            res["input_sha256"] = sha256
            tier1[ds_name] = res

    tier2_ann = run_cli_json(
        cli_path,
        [
            "ann-bench",
            "--n",
            "20000",
            "--dims",
            "128",
            "--k",
            "10",
            "--queries",
            "200",
            "--filter-pct",
            "5",
            "--seed",
            "42",
        ],
    )
    tier2_repro = run_cli_json(cli_path, ["repro-compare"])
    tier3_privacy = run_cli_json(cli_path, ["privacy-audit"])

    gates = {
        "ann_float32_recall_at_10": tier2_ann["sve0_hnsw"]["recall_at_k"] >= 0.99,
        "ann_float32_filtered_recall_at_10": (
            tier2_ann["sve0_hnsw"]["filtered_recall_at_k"] >= 0.99
        ),
        "upstream_defect_matrix": (
            tier2_repro["ghost_row_333"] == {"upstream_ghost_count": 1, "hybrid_ghost_count": 0}
            and tier2_repro["bound_int8_blob_361"] == {
                "upstream_rejected": True, "hybrid_accepted": True
            }
            and tier2_repro["l2_scale_334"]["upstream_large_is_inf"] is True
            and tier2_repro["l2_scale_334"]["upstream_tiny"] == 0
            and math.isclose(tier2_repro["l2_scale_334"]["hybrid_large"], 1e20)
            and math.isclose(tier2_repro["l2_scale_334"]["hybrid_tiny"], 1e-25)
            and tier2_repro["vec_add_overflow_363"]["upstream_silent_wrap"] == "[-106,-106]"
            and tier2_repro["vec_add_overflow_363"]["upstream_emitted"] is True
            and tier2_repro["vec_add_overflow_363"]["hybrid_rejected"] is True
        ),
        "privacy_fixtures": (
            tier3_privacy["passed"] is True
            and tier3_privacy["secret_recall"] == "15/15"
            and tier3_privacy["benign_fp"] == "0/15"
        ),
    }
    warnings = []
    if tier2_ann["sve0_hnsw_sq8"]["recall_at_k"] < 0.99:
        warnings.append("SQ8 Recall@10 is below 0.99 on the synthetic ANN fixture.")
    verdict = "FAIL" if not all(gates.values()) else ("PASS WITH WARNINGS" if warnings else "PASS")
    report = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "verified",
        "verdict": verdict,
        "gates": gates,
        "warnings": warnings,
        "methodology": {
            "tier1": "BEIR test qrels with binary relevance (positive grades); corpus-only TF-IDF/SVD",
            "tier2": "Synthetic ANN corpus and queries (seed 42); exact search as ground truth",
            "tier3": "Fixed local privacy fixtures",
        },
        "upstream_baseline": {
            "repo": "asg017/sqlite-vec",
            "commit": "04d28bd21773981e2d266bbf6aa4efbd011eb4f6",
            "version": "v0.1.10-alpha.4",
        },
        "tier1_beir_hybrid_cpu": tier1,
        "tier2_ann_scaling_20k": tier2_ann,
        "tier2_upstream_defect_matrix": tier2_repro,
        "tier3_privacy_and_dos": tier3_privacy,
    }

    with tempfile.TemporaryDirectory(prefix=".sve-report-", dir=out_path.parent) as tmp:
        staged = Path(tmp) / "report.json"
        staged.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged.replace(out_path)
    print(f"Completed three-tier benchmark: {verdict}")
    return int(verdict == "FAIL")


if __name__ == "__main__":
    raise SystemExit(main())
