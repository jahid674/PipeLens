#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_pipeline.py  (single-file RAG PipeLens-style experiment)

Goal
----
1) Use MIRACL as the source dataset.
2) Build an "English-optimized" RAG pipeline using EN corpus + EN queries (training).
3) Evaluate the same pipeline on FR queries (test) and observe failure.
4) Show that inserting a translator (FR->EN) OR switching the source corpus to FR fixes it.
5) For each historical execution (pipeline combo), store a profile vector:
   - non_english_token_ratio
   - avg_query_character_unicode_ratio
   - query_embedding_variance
   - cosine_gap
   - retrieval_variance_topk
   - retriever_metric_quality  (Recall@k if qrels exist, else pseudo)
   and save a CSV similar to your ML profile table.

Why your previous code crashed
------------------------------
MIRACL HF configs do NOT have a split named "corpus".
In the HF MIRACL loader, the *documents* are effectively in split "train" (huge),
and the queries are in small splits like "dev"/"testA"/"testB".

This file always uses:
- corpus: split="train"
- queries: split="dev" (default; can be changed)

Important runtime notes
-----------------------
- Use --smoke for small samples (fast)
- Use --run_full for larger samples (still sampled; do NOT load 32M docs)
- Uses streaming=True when reading MIRACL so we only materialize N docs.

Dependencies
------------
pip install datasets scikit-learn numpy pandas
Optional (translator):
pip install transformers sentencepiece

Examples
--------
python rag_pipeline.py --smoke
python rag_pipeline.py --run_full
python rag_pipeline.py --run_full --translator
python rag_pipeline.py --run_full --translator --fix_by_fr_corpus

Outputs
-------
historical_data/rag/
  rag_train_en_profile.csv
  rag_test_fr_profile.csv
  rag_metrics_summary.csv

"""

import os
import re
import sys
import math
import json
import time
import argparse
import logging
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("rag_pipeline")


# -----------------------------
# Utilities
# -----------------------------
def mkdirp(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def is_non_ascii(ch: str) -> bool:
    return ord(ch) > 127


def unicode_ratio(s: str) -> float:
    if not s:
        return 0.0
    c = sum(1 for ch in s if is_non_ascii(ch))
    return c / max(1, len(s))


def token_non_english_ratio(s: str) -> float:
    """
    A simple heuristic:
      - "English token" = token with only a-z letters (plus apostrophe) after lowering.
      - Non-English token = anything else (contains accents, other scripts, digits mixed, etc.)
    """
    if not s:
        return 0.0
    toks = re.findall(r"\S+", s.strip())
    if not toks:
        return 0.0
    non_en = 0
    for t in toks:
        t2 = t.lower()
        if re.fullmatch(r"[a-z']+", t2):
            continue
        non_en += 1
    return non_en / max(1, len(toks))


def stable_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.maximum(1e-12, np.sum(ex))


# -----------------------------
# MIRACL Loader
# -----------------------------
class MIRACLLoader:
    """
    Loads MIRACL EN/FR in a streaming+sampled way.

    We treat:
      - corpus = split="train"
      - queries = split="dev" by default
    because HF MIRACL scripts expose only splits: train/dev/testA/testB.

    IMPORTANT: trust_remote_code=True is required for MIRACL on HF.
    """

    def __init__(
        self,
        lang_en: str = "en",
        lang_fr: str = "fr",
        corpus_split: str = "train",
        query_split: str = "dev",
        max_docs: int = 5000,
        max_queries: int = 200,
        seed: int = 42,
        hf_home: Optional[str] = None,
    ):
        self.lang_en = lang_en
        self.lang_fr = lang_fr
        self.corpus_split = corpus_split
        self.query_split = query_split
        self.max_docs = int(max_docs)
        self.max_queries = int(max_queries)
        self.seed = int(seed)
        self.hf_home = hf_home

    def _load_dataset(self, lang: str, split: str):
        try:
            from datasets import load_dataset
        except Exception as e:
            raise RuntimeError("Missing dependency: datasets. Install with `pip install datasets`.") from e

        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home

        return load_dataset(
            "miracl/miracl",
            lang,
            split=split,
            streaming=True,
            trust_remote_code=True,
        )

    def _take_n_stream(self, ds_stream, n: int) -> List[dict]:
        out = []
        for i, ex in enumerate(ds_stream):
            out.append(ex)
            if (i + 1) >= n:
                break
        return out

    def _infer_doc_fields(self, ex: dict) -> Tuple[str, str]:
        """
        Return (doc_id, doc_text) from a MIRACL corpus row.
        Make robust to different field names.
        """
        # Common patterns seen in MIRACL-like corpora:
        # id / docid, title, text, contents, body, etc.
        doc_id = None
        for k in ["docid", "doc_id", "id", "_id"]:
            if k in ex:
                doc_id = str(ex[k])
                break
        if doc_id is None:
            doc_id = str(ex.get("pid", ex.get("passage_id", "")))

        title = ex.get("title", "")
        text = ex.get("text", ex.get("contents", ex.get("body", "")))
        doc_text = (str(title) + " " + str(text)).strip()
        return doc_id, doc_text

    def _infer_query_fields(self, ex: dict) -> Tuple[str, str, List[str]]:
        """
        Return (qid, query, relevant_doc_ids_if_any) from a MIRACL query row.

        Some scripts include qrels-like info in fields such as:
          - 'positive_passages' (list of dicts with 'docid'/'passage_id')
          - 'answers' etc.

        If no relevance ids are available, we return [].
        """
        qid = None
        for k in ["query_id", "qid", "id"]:
            if k in ex:
                qid = str(ex[k])
                break
        if qid is None:
            qid = str(ex.get("_id", ""))

        query = ex.get("query", ex.get("question", ex.get("text", "")))
        query = str(query)

        rel_ids: List[str] = []

        # Possible relevance fields
        if "positive_passages" in ex and isinstance(ex["positive_passages"], list):
            for p in ex["positive_passages"]:
                if isinstance(p, dict):
                    for k in ["docid", "doc_id", "id", "passage_id", "pid"]:
                        if k in p:
                            rel_ids.append(str(p[k]))
                            break

        if "positive_passage_ids" in ex and isinstance(ex["positive_passage_ids"], list):
            rel_ids.extend([str(x) for x in ex["positive_passage_ids"]])

        if "relevant_docids" in ex and isinstance(ex["relevant_docids"], list):
            rel_ids.extend([str(x) for x in ex["relevant_docids"]])

        # Dedup preserve order
        seen = set()
        rel_ids_uniq = []
        for rid in rel_ids:
            if rid not in seen and rid != "":
                seen.add(rid)
                rel_ids_uniq.append(rid)

        return qid, query, rel_ids_uniq

    def load_en_fr(self):
        LOGGER.info("[LOAD] Loading MIRACL EN/FR (streaming; sampled)...")

        # Corpus (documents) from split="train"
        corp_en_stream = self._load_dataset(self.lang_en, self.corpus_split)
        corp_fr_stream = self._load_dataset(self.lang_fr, self.corpus_split)

        corp_en_raw = self._take_n_stream(corp_en_stream, self.max_docs)
        corp_fr_raw = self._take_n_stream(corp_fr_stream, self.max_docs)

        corp_en = []
        for ex in corp_en_raw:
            did, txt = self._infer_doc_fields(ex)
            if txt:
                corp_en.append({"doc_id": did, "text": txt})
        corp_fr = []
        for ex in corp_fr_raw:
            did, txt = self._infer_doc_fields(ex)
            if txt:
                corp_fr.append({"doc_id": did, "text": txt})

        # Queries from split="dev" (small)
        q_en_stream = self._load_dataset(self.lang_en, self.query_split)
        q_fr_stream = self._load_dataset(self.lang_fr, self.query_split)

        q_en_raw = self._take_n_stream(q_en_stream, self.max_queries)
        q_fr_raw = self._take_n_stream(q_fr_stream, self.max_queries)

        q_en = []
        qrels_en = {}  # qid -> set(doc_id)
        for ex in q_en_raw:
            qid, q, rel_ids = self._infer_query_fields(ex)
            if q.strip():
                q_en.append({"qid": qid, "query": q})
                if rel_ids:
                    qrels_en[qid] = set(rel_ids)

        q_fr = []
        qrels_fr = {}
        for ex in q_fr_raw:
            qid, q, rel_ids = self._infer_query_fields(ex)
            if q.strip():
                q_fr.append({"qid": qid, "query": q})
                if rel_ids:
                    qrels_fr[qid] = set(rel_ids)

        corp_en_df = pd.DataFrame(corp_en)
        corp_fr_df = pd.DataFrame(corp_fr)
        q_en_df = pd.DataFrame(q_en)
        q_fr_df = pd.DataFrame(q_fr)

        LOGGER.info(f"[LOAD] EN corpus docs: {len(corp_en_df)} | EN queries: {len(q_en_df)}")
        LOGGER.info(f"[LOAD] FR corpus docs: {len(corp_fr_df)} | FR queries: {len(q_fr_df)}")
        LOGGER.info(f"[LOAD] qrels_en available for {len(qrels_en)} queries; qrels_fr for {len(qrels_fr)} queries.")

        return corp_en_df, corp_fr_df, q_en_df, q_fr_df, qrels_en, qrels_fr


# -----------------------------
# Translator (optional)
# -----------------------------
class FR2ENTranslator:
    """
    Optional translator using HF MarianMT.
    If transformers is missing or model download fails, it gracefully falls back to identity.
    """

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self.ready = False
        self.tokenizer = None
        self.model = None

        if not self.enabled:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            name = "Helsinki-NLP/opus-mt-fr-en"
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(name)
            self.ready = True
            LOGGER.info("[TRANSLATOR] Loaded Helsinki-NLP/opus-mt-fr-en")
        except Exception as e:
            self.ready = False
            LOGGER.warning(f"[TRANSLATOR] Could not load translator model. Falling back to identity. Error: {e}")

    def translate(self, text: str) -> str:
        if not self.enabled or not self.ready:
            return text
        try:
            import torch
            inputs = self.tokenizer([text], return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=256, num_beams=4)
            return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        except Exception:
            return text


# -----------------------------
# RAG Components (simple + robust)
# -----------------------------
class QueryNormalizer:
    def __init__(self, mode: str):
        self.mode = mode

    def apply(self, q: str) -> str:
        q2 = q
        if self.mode == "none":
            return q2
        if self.mode == "lower+strip_punct":
            q2 = q2.lower()
            q2 = re.sub(r"[^\w\s]", " ", q2, flags=re.UNICODE)
            q2 = re.sub(r"\s+", " ", q2).strip()
            return q2
        if self.mode == "spellfix_light":
            # Very light normalization: lowercase + collapse repeated chars (toy "spellfix")
            q2 = q2.lower()
            q2 = re.sub(r"(.)\1{2,}", r"\1\1", q2)
            q2 = re.sub(r"\s+", " ", q2).strip()
            return q2
        return q2


class Retriever:
    """
    Retrieval over a corpus with:
      - "bm25@k"     -> word TF-IDF cosine (works like sparse)
      - "dense@k"    -> char ngram TF-IDF cosine (more robust across typos)
      - "hybrid_rrf@k" -> RRF fusion of both rankings
    """
    def __init__(self, mode: str, corpus_texts: List[str], corpus_ids: List[str]):
        self.mode = mode
        self.corpus_texts = corpus_texts
        self.corpus_ids = corpus_ids

        # build vectorizers lazily (only if needed)
        self.word_vec = None
        self.word_X = None
        self.char_vec = None
        self.char_X = None

        self._ensure_indexes()

    def _ensure_indexes(self):
        if self.mode.startswith("bm25") or self.mode.startswith("hybrid"):
            self.word_vec = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=200000,
                ngram_range=(1, 2),
            )
            self.word_X = self.word_vec.fit_transform(self.corpus_texts)
            self.word_X = normalize(self.word_X)

        if self.mode.startswith("dense") or self.mode.startswith("hybrid"):
            self.char_vec = TfidfVectorizer(
                lowercase=True,
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=200000,
            )
            self.char_X = self.char_vec.fit_transform(self.corpus_texts)
            self.char_X = normalize(self.char_X)

    def _parse_k(self) -> int:
        # format: "bm25@10"
        try:
            return int(self.mode.split("@")[1])
        except Exception:
            return 10

    def retrieve(self, query: str) -> Tuple[List[str], List[float]]:
        k = self._parse_k()
        if self.mode.startswith("bm25"):
            qv = self.word_vec.transform([query])
            qv = normalize(qv)
            sims = (self.word_X @ qv.T).toarray().ravel()
            idx = np.argsort(-sims)[:k]
            return [self.corpus_ids[i] for i in idx], [float(sims[i]) for i in idx]

        if self.mode.startswith("dense"):
            qv = self.char_vec.transform([query])
            qv = normalize(qv)
            sims = (self.char_X @ qv.T).toarray().ravel()
            idx = np.argsort(-sims)[:k]
            return [self.corpus_ids[i] for i in idx], [float(sims[i]) for i in idx]

        if self.mode.startswith("hybrid"):
            # RRF fusion of word + char rankings
            qv_w = normalize(self.word_vec.transform([query]))
            sims_w = (self.word_X @ qv_w.T).toarray().ravel()
            rank_w = np.argsort(-sims_w)

            qv_c = normalize(self.char_vec.transform([query]))
            sims_c = (self.char_X @ qv_c.T).toarray().ravel()
            rank_c = np.argsort(-sims_c)

            # RRF
            K = 60.0
            rrf = np.zeros(len(self.corpus_ids), dtype=float)

            # only need top ~2000 from each rank for speed
            topN = min(2000, len(self.corpus_ids))
            inv_pos = np.empty(len(self.corpus_ids), dtype=int)
            inv_pos[rank_w] = np.arange(len(rank_w))
            inv_pos2 = np.empty(len(self.corpus_ids), dtype=int)
            inv_pos2[rank_c] = np.arange(len(rank_c))

            idx_w = rank_w[:topN]
            idx_c = rank_c[:topN]
            rrf[idx_w] += 1.0 / (K + inv_pos[idx_w] + 1.0)
            rrf[idx_c] += 1.0 / (K + inv_pos2[idx_c] + 1.0)

            idx = np.argsort(-rrf)[:k]
            # score: use rrf as "score"
            return [self.corpus_ids[i] for i in idx], [float(rrf[i]) for i in idx]

        # fallback
        return [], []


class Reranker:
    """
    Simple reranker:
      - none
      - cross_encoder@20 / @50 (simulated by re-scoring with word tfidf similarity)
    """
    def __init__(self, mode: str, corpus_map: Dict[str, str]):
        self.mode = mode
        self.corpus_map = corpus_map
        self.vec = TfidfVectorizer(lowercase=True, stop_words="english", max_features=100000, ngram_range=(1, 2))
        # Fit later per batch for small rerank sets is ok.

    def _parse_k(self) -> int:
        try:
            return int(self.mode.split("@")[1])
        except Exception:
            return 20

    def rerank(self, query: str, doc_ids: List[str]) -> List[str]:
        if self.mode == "none" or not doc_ids:
            return doc_ids
        k = min(self._parse_k(), len(doc_ids))
        # "cross-encoder" simulation: fit tfidf on query+docs and sort by cosine
        docs = [self.corpus_map[i] for i in doc_ids[:k]]
        X = self.vec.fit_transform([query] + docs)
        qv = normalize(X[0])
        dv = normalize(X[1:])
        sims = (dv @ qv.T).toarray().ravel()
        order = np.argsort(-sims)
        reranked = [doc_ids[i] for i in order]
        # keep tail (if any)
        return reranked + doc_ids[k:]


class ContextBuilder:
    """
    Build context from top docs:
      - fixed_chunk_top@1500 / @3000 (concat until char limit)
      - mmr_diverse@1500 / @3000 (simple diversity by penalizing near-duplicate docs)
    """
    def __init__(self, mode: str, corpus_map: Dict[str, str]):
        self.mode = mode
        self.corpus_map = corpus_map

    def _parse_limit(self) -> int:
        try:
            return int(self.mode.split("@")[1])
        except Exception:
            return 1500

    def build(self, doc_ids: List[str]) -> str:
        if not doc_ids:
            return ""
        limit = self._parse_limit()
        if self.mode.startswith("fixed_chunk_top"):
            out = []
            total = 0
            for did in doc_ids:
                txt = self.corpus_map[did]
                if total + len(txt) > limit:
                    txt = txt[: max(0, limit - total)]
                if txt:
                    out.append(txt)
                    total += len(txt)
                if total >= limit:
                    break
            return "\n\n".join(out)

        if self.mode.startswith("mmr_diverse"):
            # diversity: avoid adding if too similar (by simple substring overlap heuristic)
            out = []
            total = 0
            chosen = []
            for did in doc_ids:
                txt = self.corpus_map[did]
                if not txt:
                    continue
                ok = True
                for c in chosen:
                    # crude overlap test
                    if len(set(txt.split()) & set(c.split())) / max(1, len(set(txt.split()))) > 0.6:
                        ok = False
                        break
                if not ok:
                    continue
                if total + len(txt) > limit:
                    txt = txt[: max(0, limit - total)]
                if txt:
                    out.append(txt)
                    chosen.append(txt)
                    total += len(txt)
                if total >= limit:
                    break
            return "\n\n".join(out)

        return ""


class Generator:
    """
    Not a real LLM. We just produce a deterministic "answer"
    so that the pipeline has an end-to-end output.
    """
    def __init__(self, mode: str):
        self.mode = mode

    def generate(self, query: str, context: str) -> str:
        if self.mode == "grounded_concise":
            return f"Q: {query}\nA (from context): {context[:300]}"
        return f"Q: {query}\nA (from context): {context[:800]}"


# -----------------------------
# Profiles
# -----------------------------
@dataclass
class QueryProfile:
    non_english_token_ratio: float
    avg_unicode_ratio: float
    embedding_variance: float
    cosine_gap: float
    retrieval_variance_topk: float
    retriever_metric_quality: float  # recall@k if qrels known else pseudo


class RAGProfiler:
    """
    Computes the requested profiles for a dataset run.

    "Embedding" is approximated as the TF-IDF query vector projected into a smaller space
    (we just use the normalized TF-IDF vector directly and compute variance).
    """

    def __init__(self, topk: int):
        self.topk = int(topk)

    def compute_per_query(
        self,
        query: str,
        q_vec_dense: np.ndarray,
        retrieved_scores: List[float],
        rel_ids: Optional[set],
        retrieved_doc_ids: List[str],
    ) -> QueryProfile:
        # 1) language-ish profiles
        ner = token_non_english_ratio(query)
        ur = unicode_ratio(query)

        # 2) "embedding variance": variance of the dense query representation
        emb_var = float(np.var(q_vec_dense)) if q_vec_dense.size else 0.0

        # 3) cosine gap: top1 - mean(topk)
        if retrieved_scores:
            top1 = float(retrieved_scores[0])
            mean_k = float(np.mean(retrieved_scores[: min(self.topk, len(retrieved_scores))]))
            cos_gap = top1 - mean_k
        else:
            cos_gap = 0.0

        # 4) retrieval variance topk
        if retrieved_scores:
            rv = float(np.var(retrieved_scores[: min(self.topk, len(retrieved_scores))]))
        else:
            rv = 0.0

        # 5) retriever metric quality: recall@k if qrels provided
        # If no qrels, use a pseudo score based on score mass in topk.
        if rel_ids is not None and len(rel_ids) > 0:
            got = set(retrieved_doc_ids[: min(self.topk, len(retrieved_doc_ids))])
            rec = len(got & rel_ids) / max(1, len(rel_ids))
            qual = float(rec)
        else:
            # pseudo quality: softmax mass of topk (higher concentration -> "confident retrieval")
            if retrieved_scores:
                s = np.array(retrieved_scores[: min(self.topk, len(retrieved_scores))], dtype=float)
                p = stable_softmax(s)
                qual = float(np.max(p))
            else:
                qual = 0.0

        return QueryProfile(
            non_english_token_ratio=float(ner),
            avg_unicode_ratio=float(ur),
            embedding_variance=float(emb_var),
            cosine_gap=float(cos_gap),
            retrieval_variance_topk=float(rv),
            retriever_metric_quality=float(qual),
        )

    def summarize(self, profiles: List[QueryProfile]) -> Dict[str, float]:
        if not profiles:
            return {
                "non_english_token_ratio": 0.0,
                "avg_query_character_unicode_ratio": 0.0,
                "query_embedding_variance": 0.0,
                "cosine_gap": 0.0,
                "retrieval_variance_topk": 0.0,
                "retriever_metric_quality": 0.0,
            }
        df = pd.DataFrame([p.__dict__ for p in profiles])
        return {
            "non_english_token_ratio": float(df["non_english_token_ratio"].mean()),
            "avg_query_character_unicode_ratio": float(df["avg_unicode_ratio"].mean()),
            "query_embedding_variance": float(df["embedding_variance"].mean()),
            "cosine_gap": float(df["cosine_gap"].mean()),
            "retrieval_variance_topk": float(df["retrieval_variance_topk"].mean()),
            "retriever_metric_quality": float(df["retriever_metric_quality"].mean()),
        }


# -----------------------------
# RAG Pipeline Executor (PipeLens-style table)
# -----------------------------
class RAGPipelineExecutor:
    """
    Enumerates pipeline combos (like your run_pipeline_glass_sample) and stores:
      [pipeline params] + [profiles] + [utility]

    Utility here is: 1 - Recall@k  (lower is better), if qrels exist;
    else utility is: 1 - pseudo_quality
    """

    def __init__(
        self,
        corpus_df: pd.DataFrame,
        queries_df: pd.DataFrame,
        qrels: Dict[str, set],
        translator: FR2ENTranslator,
        execution_name: str,
        out_dir: str,
        seed: int = 42,
    ):
        self.corpus_df = corpus_df
        self.queries_df = queries_df
        self.qrels = qrels or {}
        self.translator = translator
        self.execution_name = execution_name
        self.out_dir = out_dir
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

        # Pipeline strategies (match your idea)
        self.query_normalizer_strategy = ["none", "lower+strip_punct", "spellfix_light"]
        self.retriever_strategy = ["bm25@10", "bm25@20", "dense@10", "dense@20", "hybrid_rrf@10", "hybrid_rrf@20"]
        self.reranker_strategy = ["none", "cross_encoder@20", "cross_encoder@50"]
        self.context_builder_strategy = ["fixed_chunk_top@1500", "fixed_chunk_top@3000", "mmr_diverse@1500", "mmr_diverse@3000"]
        self.generator_strategy = ["grounded_concise", "grounded_detailed"]

        # Insertion component strategies
        self.translator_rewriter_strategy = ["mt_generic|fr->en|literal"]

        # Canonical base pipeline order
        self.pipeline_order = ["query_normalize", "retriever", "reranker", "context_builder", "generator"]

        # Build corpus structures
        self.corpus_ids = self.corpus_df["doc_id"].astype(str).tolist()
        self.corpus_texts = self.corpus_df["text"].astype(str).tolist()
        self.corpus_map = dict(zip(self.corpus_ids, self.corpus_texts))

    def _strategy_counts(self) -> Dict[str, int]:
        return {
            "query_normalize": len(self.query_normalizer_strategy),
            "retriever": len(self.retriever_strategy),
            "reranker": len(self.reranker_strategy),
            "context_builder": len(self.context_builder_strategy),
            "generator": len(self.generator_strategy),
            "translator_rewriter": len(self.translator_rewriter_strategy),
        }

    def _safe_param_index(self, step: str, raw_value: int) -> int:
        counts = self._strategy_counts()
        n = counts.get(step, 0)
        if n <= 0:
            raise ValueError(f"Unknown step '{step}'")
        v = int(raw_value)
        if 1 <= v <= n:
            return v - 1
        if 0 <= v < n:
            return v
        raise ValueError(f"Param out of range for step={step}: got {v}, valid 1..{n} or 0..{n-1}")

    def _build_pipeline_objects(
        self,
        order: List[str],
        params_1b: List[int],
        use_translator: bool,
    ):
        # params_1b aligned with order
        # translator affects query before normalization or after? We'll do: translate FIRST, then normalize.
        # If inserted, we treat it as its own component in `order`.
        objs = {}
        for step, p1 in zip(order, params_1b):
            idx = self._safe_param_index(step, p1)
            if step == "query_normalize":
                objs[step] = QueryNormalizer(self.query_normalizer_strategy[idx])
            elif step == "retriever":
                mode = self.retriever_strategy[idx]
                objs[step] = Retriever(mode, self.corpus_texts, self.corpus_ids)
            elif step == "reranker":
                objs[step] = Reranker(self.reranker_strategy[idx], self.corpus_map)
            elif step == "context_builder":
                objs[step] = ContextBuilder(self.context_builder_strategy[idx], self.corpus_map)
            elif step == "generator":
                objs[step] = Generator(self.generator_strategy[idx])
            elif step == "translator_rewriter":
                # just a marker; actual translation uses self.translator
                objs[step] = "translator_marker"
            else:
                raise ValueError(f"Unknown step in order: {step}")

        return objs

    def _run_single_query(
        self,
        pipeline_order: List[str],
        pipeline_params_1b: List[int],
        qid: str,
        query: str,
        rel_ids: Optional[set],
        topk_for_profiles: int,
        use_translator_component: bool,
    ) -> Tuple[QueryProfile, float]:
        """
        Returns:
          - query profile
          - query utility (lower is better)
        """
        objs = self._build_pipeline_objects(pipeline_order, pipeline_params_1b, use_translator_component)

        q = query

        # If translator component is present, translate before normalization
        if use_translator_component and "translator_rewriter" in pipeline_order:
            q = self.translator.translate(q)

        # normalize
        q = objs["query_normalize"].apply(q)

        # retrieve
        doc_ids, scores = objs["retriever"].retrieve(q)

        # rerank
        doc_ids = objs["reranker"].rerank(q, doc_ids)

        # rebuild scores list after rerank: approximate by re-scoring using position
        # (for profiling we mainly need topk stats; keep original scores if lengths match)
        if len(scores) != len(doc_ids):
            # if rerank changed order, make scores monotone decreasing proxy
            scores = [1.0 / (i + 1) for i in range(len(doc_ids))]

        # context
        ctx = objs["context_builder"].build(doc_ids)

        # generate (not used for utility)
        _ = objs["generator"].generate(q, ctx)

        # Build a "dense" embedding for the query for embedding variance:
        # Use char tfidf (language-robust) as pseudo-embedding.
        # (Fit on-the-fly for single query is OK because we only use variance; deterministic.)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=5000)
        Xq = vec.fit_transform([q])
        Xq = normalize(Xq).toarray().ravel()

        profiler = RAGProfiler(topk=topk_for_profiles)
        qp = profiler.compute_per_query(
            query=query,
            q_vec_dense=Xq,
            retrieved_scores=scores,
            rel_ids=rel_ids,
            retrieved_doc_ids=doc_ids,
        )

        # Utility:
        # If we have rel_ids, use (1 - recall@k); else use (1 - pseudo_quality)
        util = 1.0 - float(qp.retriever_metric_quality)
        return qp, util

    def run_glass_sample(
        self,
        out_csv: str,
        n_samples: int,
        seed: int,
        use_translator_component: bool = False,
        insert_pos: int = 1,  # translator inserted after query_normalize? We'll do pos=1 means after query_normalize
        topk_for_profiles: int = 10,
    ) -> pd.DataFrame:
        """
        Similar to your run_pipeline_glass_sample:
        - Sample random pipeline combos
        - For each combo, compute mean profiles over all queries (or a sampled subset)
        - Save: params + profiles + utility

        If use_translator_component=True:
          we insert "translator_rewriter" into pipeline at `insert_pos`.
        """
        mkdirp(os.path.dirname(out_csv))
        if os.path.exists(out_csv):
            LOGGER.info(f"[RAG] Found existing file -> load: {out_csv}")
            return pd.read_csv(out_csv)

        rng = np.random.default_rng(seed)
        counts = self._strategy_counts()

        base_order = self.pipeline_order[:]
        if use_translator_component:
            if insert_pos < 0:
                insert_pos = 0
            if insert_pos > len(base_order):
                insert_pos = len(base_order)
            order = base_order[:insert_pos] + ["translator_rewriter"] + base_order[insert_pos:]
        else:
            order = base_order

        # build sampling space sizes
        sizes = [counts[s] for s in order]
        total_space = int(np.prod(sizes)) if sizes else 0
        n = int(min(max(1, n_samples), max(1, total_space)))

        # sample unique combinations (0-based internally; store 1-based)
        sampled = []
        seen = set()
        attempts = 0
        max_attempts = 10 * n + 10000

        while len(sampled) < n and attempts < max_attempts:
            attempts += 1
            combo0 = [int(rng.integers(0, sizes[i])) for i in range(len(sizes))]
            combo1 = tuple([c + 1 for c in combo0])
            if combo1 in seen:
                continue
            seen.add(combo1)
            sampled.append(list(combo1))

        # choose query subset (keep stable)
        Q = self.queries_df.copy()
        if len(Q) == 0:
            raise RuntimeError("No queries available.")
        # for speed, limit to <= 200 queries
        max_eval_q = min(200, len(Q))
        q_idx = rng.choice(len(Q), size=max_eval_q, replace=False)
        Q = Q.iloc[q_idx].reset_index(drop=True)

        rows = []
        profile_cols = [
            "non_english_token_ratio",
            "avg_query_character_unicode_ratio",
            "query_embedding_variance",
            "cosine_gap",
            "retrieval_variance_topk",
            "retriever_metric_quality",
        ]
        utility_col = "utility_rag"

        LOGGER.info(f"[RAG] Evaluating {len(sampled)} pipeline combos on {len(Q)} queries. translator={use_translator_component}")

        for params_1b in sampled:
            qprofiles: List[QueryProfile] = []
            utils: List[float] = []

            for _, r in Q.iterrows():
                qid = str(r["qid"])
                query = str(r["query"])
                rel = self.qrels.get(qid, None)
                qp, u = self._run_single_query(
                    pipeline_order=order,
                    pipeline_params_1b=params_1b,
                    qid=qid,
                    query=query,
                    rel_ids=rel,
                    topk_for_profiles=topk_for_profiles,
                    use_translator_component=use_translator_component,
                )
                qprofiles.append(qp)
                utils.append(u)

            prof = RAGProfiler(topk=topk_for_profiles).summarize(qprofiles)
            utility = float(np.mean(utils)) if utils else 1.0

            row = params_1b + [prof[c] for c in profile_cols] + [utility]
            rows.append(row)

        cols = order + profile_cols + [utility_col]
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(out_csv, index=False)
        LOGGER.info(f"[RAG] Wrote: {out_csv} ({len(df)} rows)")
        return df


# -----------------------------
# Experiment runner
# -----------------------------
def run_full_experiment(
    smoke: bool,
    run_full: bool,
    use_translator_fix: bool,
    fix_by_fr_corpus: bool,
    out_dir: str = "historical_data/rag",
    seed: int = 42,
):
    """
    Runs 3 evaluations:

    A) Train (EN corpus, EN queries) generate historical table
    B) Test  (EN corpus, FR queries) same pipeline space -> expected failure patterns
    C) Fix:
       - If --translator: insert translator component
       - If --fix_by_fr_corpus: use FR corpus with FR queries (no translator)

    Produces:
      rag_train_en_profile.csv
      rag_test_fr_profile.csv
      rag_fix_profile.csv
      rag_metrics_summary.csv
    """
    mkdirp(out_dir)

    # sampling knobs
    if smoke:
        max_docs = 4000
        max_queries = 80
        n_samples = 50
        insert_pos = 1
    else:
        max_docs = 20000 if run_full else 8000
        max_queries = 200 if run_full else 120
        n_samples = 400 if run_full else 120
        insert_pos = 1

    loader = MIRACLLoader(
        corpus_split="train",
        query_split="dev",
        max_docs=max_docs,
        max_queries=max_queries,
        seed=seed,
        hf_home=os.environ.get("HF_HOME", None),
    )

    corp_en, corp_fr, q_en, q_fr, qrels_en, qrels_fr = loader.load_en_fr()

    # translator
    translator = FR2ENTranslator(enabled=use_translator_fix)

    # -------- A) training (EN corpus + EN queries)
    train_csv = os.path.join(out_dir, "rag_train_en_profile.csv")
    exec_train = RAGPipelineExecutor(
        corpus_df=corp_en,
        queries_df=q_en,
        qrels=qrels_en,
        translator=translator,
        execution_name="train_en",
        out_dir=out_dir,
        seed=seed,
    )
    df_train = exec_train.run_glass_sample(
        out_csv=train_csv,
        n_samples=n_samples,
        seed=seed,
        use_translator_component=False,
        insert_pos=insert_pos,
        topk_for_profiles=10,
    )

    # Find best EN pipeline (lowest utility)
    best_train_row = df_train.iloc[int(df_train["utility_rag"].astype(float).idxmin())]
    best_train_params = [int(best_train_row[c]) for c in exec_train.pipeline_order]

    # -------- B) test failure (EN corpus + FR queries; NO translator)
    test_csv = os.path.join(out_dir, "rag_test_fr_profile.csv")
    exec_test = RAGPipelineExecutor(
        corpus_df=corp_en,         # keep EN corpus to force failure on FR queries
        queries_df=q_fr,
        qrels=qrels_fr,
        translator=translator,
        execution_name="test_fr",
        out_dir=out_dir,
        seed=seed,
    )
    df_test = exec_test.run_glass_sample(
        out_csv=test_csv,
        n_samples=n_samples,
        seed=seed + 1,
        use_translator_component=False,
        insert_pos=insert_pos,
        topk_for_profiles=10,
    )

    # Evaluate the EN-best pipeline directly on FR queries (diagnostic)
    # We'll compute its profiles+utility as a single row
    def eval_single(executor: RAGPipelineExecutor, order: List[str], params: List[int], use_translator_component: bool):
        Q = executor.queries_df.copy().reset_index(drop=True)
        Q = Q.iloc[: min(200, len(Q))].reset_index(drop=True)
        qprofiles, utils = [], []
        for _, r in Q.iterrows():
            qid = str(r["qid"])
            query = str(r["query"])
            rel = executor.qrels.get(qid, None)
            qp, u = executor._run_single_query(
                pipeline_order=order,
                pipeline_params_1b=params,
                qid=qid,
                query=query,
                rel_ids=rel,
                topk_for_profiles=10,
                use_translator_component=use_translator_component,
            )
            qprofiles.append(qp)
            utils.append(u)
        prof = RAGProfiler(topk=10).summarize(qprofiles)
        return prof, float(np.mean(utils)) if utils else 1.0

    base_order = exec_train.pipeline_order[:]  # 5 steps
    prof_fr_fail, util_fr_fail = eval_single(exec_test, base_order, best_train_params, use_translator_component=False)

    # -------- C) FIX path
    fix_csv = os.path.join(out_dir, "rag_fix_profile.csv")

    if fix_by_fr_corpus:
        # Switch to FR corpus, FR queries (no translator)
        exec_fix = RAGPipelineExecutor(
            corpus_df=corp_fr,
            queries_df=q_fr,
            qrels=qrels_fr,
            translator=translator,
            execution_name="fix_fr_corpus",
            out_dir=out_dir,
            seed=seed,
        )
        df_fix = exec_fix.run_glass_sample(
            out_csv=fix_csv,
            n_samples=n_samples,
            seed=seed + 2,
            use_translator_component=False,
            insert_pos=insert_pos,
            topk_for_profiles=10,
        )
        # Evaluate EN-best params on FR corpus (same order)
        prof_fr_fix, util_fr_fix = eval_single(exec_fix, base_order, best_train_params, use_translator_component=False)

        fix_mode = "fr_corpus"
        fix_util = util_fr_fix
        fix_prof = prof_fr_fix

    elif use_translator_fix:
        # Keep EN corpus, FR queries, but insert translator component
        exec_fix = RAGPipelineExecutor(
            corpus_df=corp_en,
            queries_df=q_fr,
            qrels=qrels_fr,
            translator=translator,
            execution_name="fix_translator",
            out_dir=out_dir,
            seed=seed,
        )
        # We will generate a historical table for the pipeline with translator inserted
        df_fix = exec_fix.run_glass_sample(
            out_csv=fix_csv,
            n_samples=n_samples,
            seed=seed + 2,
            use_translator_component=True,
            insert_pos=insert_pos,  # translator inserted after query_normalize (pos=1)
            topk_for_profiles=10,
        )
        # Evaluate EN-best params with translator inserted:
        # If translator inserted, order changes; params must include translator param.
        # We'll use translator param = 1 (only strategy).
        fix_order = base_order[:insert_pos] + ["translator_rewriter"] + base_order[insert_pos:]
        fix_params = best_train_params[:insert_pos] + [1] + best_train_params[insert_pos:]
        prof_fr_fix, util_fr_fix = eval_single(exec_fix, fix_order, fix_params, use_translator_component=True)

        fix_mode = "translator"
        fix_util = util_fr_fix
        fix_prof = prof_fr_fix

    else:
        # No fix requested: still write summary
        df_fix = pd.DataFrame([])
        fix_mode = "none"
        fix_util = util_fr_fail
        fix_prof = prof_fr_fail

    # -------- Summary report
    summary = {
        "train_best_utility_en": float(best_train_row["utility_rag"]),
        "test_fr_utility_with_en_best_pipeline": float(util_fr_fail),
        "fix_mode": fix_mode,
        "fix_fr_utility": float(fix_util),
        # delta (smaller is better; utility = 1 - quality)
        "delta_fail_minus_fix": float(util_fr_fail - fix_util),
    }

    # attach key profile means for fail vs fix
    for k, v in prof_fr_fail.items():
        summary[f"fail_fr_{k}"] = float(v)
    for k, v in fix_prof.items():
        summary[f"fix_fr_{k}"] = float(v)

    summary_csv = os.path.join(out_dir, "rag_metrics_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    LOGGER.info("========================================")
    LOGGER.info("[RESULT] Summary")
    for k in summary:
        LOGGER.info(f"  {k}: {summary[k]}")
    LOGGER.info(f"[RESULT] Wrote: {summary_csv}")
    LOGGER.info("========================================")


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="small/faster run")
    ap.add_argument("--run_full", action="store_true", help="larger sampled run")
    ap.add_argument("--translator", action="store_true", help="enable FR->EN translator fix (insertion)")
    ap.add_argument("--fix_by_fr_corpus", action="store_true", help="fix by switching source corpus to FR instead")
    ap.add_argument("--out_dir", type=str, default="historical_data/rag")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.smoke and not args.run_full:
        print("Nothing to do. Use one of: --smoke or --run_full")
        print("Example:")
        print("  python rag_pipeline.py --smoke")
        print("  python rag_pipeline.py --run_full --translator")
        sys.exit(0)

    run_full_experiment(
        smoke=args.smoke,
        run_full=args.run_full,
        use_translator_fix=args.translator,
        fix_by_fr_corpus=args.fix_by_fr_corpus,
        out_dir=args.out_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
