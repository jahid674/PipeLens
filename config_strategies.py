# config_strategies.py
"""Single source of truth for strategies & shared config (keys match handlers)."""

def build_strategies(pipeline_type: str = "ml") -> dict:
    if pipeline_type != "ml":
        return {}
    return {
        # structured
        "mv_strategy": ["drop", "mean", "median", "most_frequent", "knn"],
        "norm_strategy": ["none", "ss", "rs", "ma", "mm"],
        "od_strategy": ["none", "if", "iqr", "lof"],
        "model_selection": ["lr"],
        "knn_k_lst": [1, 5, 10, 20, 30],
        "lof_k_lst": [1, 5, 10, 20, 30],
        # text
        "pr_strategy": ["none", "pr"],
        "lowercase_strategy": ["none", "lc"],
        "spellcheck_strategy": ["none", "sc"],
        "whitespace_strategy": ["none", "wc"],
        "unit_converter_strategy": ["none", "uc"],
        "tokenization_strategy": ["none", "whitespace", "nltk"],
        "stopword_strategy": ["none", "sw"],
        "specialchar_strategy": ["none", "sc"],
        "deduplication_strategy": ["none", "dd"],
        "fselection_strategy": ["none", "variance", "mutual_info"],
        "language_translator_strategy": ["none", "lt"],
        "language_detector_strategy":["none", "ld"]
    }

def build_strategy_counts(s: dict) -> dict:
    """Counts keyed by pipeline steps; expands KNN/LOF like original."""
    mv, od = s["mv_strategy"], s["od_strategy"]
    return {
        "missing_value": len(mv) + (len(s["knn_k_lst"]) - 1 if "knn" in mv else 0),
        "normalization": len(s["norm_strategy"]),
        "outlier": len(od) + (len(s["lof_k_lst"]) - 1 if "lof" in od else 0),
        "model": len(s["model_selection"]),
        "punctuation": len(s["pr_strategy"]),
        "whitespace": len(s["whitespace_strategy"]),
        "unit_converter": len(s["unit_converter_strategy"]),
        "tokenizer": len(s["tokenization_strategy"]),
        "stopword": len(s["stopword_strategy"]),
        "spell_checker": len(s["spellcheck_strategy"]),
        "special_character": len(s["specialchar_strategy"]),
        "deduplication": len(s["deduplication_strategy"]),
        "fselection":len(s["fselection_strategy"]),
        "language_translator":len(s["language_translator_strategy"]),
        "language_detector":len(s["language_detector_strategy"])

    }

def build_shared_config(s: dict, metric_type: str, sensitive_var: str, target_var: str) -> dict:
    """Exact keys used by handlers; thresholds filled by executor later."""
    return {
        # structured
        "mv_strategy": s["mv_strategy"],
        "knn_k_lst": s["knn_k_lst"],
        "norm_strategy": s["norm_strategy"],
        "od_strategy": s["od_strategy"],
        "lof_k_lst": s["lof_k_lst"],
        "model_selection": s["model_selection"],
        # globals
        "metric_type": metric_type,
        "sensitive_var": sensitive_var,
        "target_var": target_var,
        # text
        "punctuation_strategy": s["pr_strategy"],
        "whitespace_strategy": s["whitespace_strategy"],
        "unit_converter_strategy": s["unit_converter_strategy"],
        "tokenization_strategy": s["tokenization_strategy"],
        "stopword_strategy": s["stopword_strategy"],
        "spellchecker_strategy": s["spellcheck_strategy"],
        "specialchar_strategy": s["specialchar_strategy"],
        "deduplication_strategy": s["deduplication_strategy"],
        "fselection_stategy":s["fselection_strategy"],
        "language_translator_strategy":s["language_translator_strategy"],
        "language_detector_strategy":s["language_detector_strategy"],
        # thresholds injected later
        "contamination": None,
        "contamination_lof": None,
    }
