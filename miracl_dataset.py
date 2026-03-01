import pandas as pd
from datasets import load_dataset

def load_miracl(lang="en", split="train", max_rows=None):
    """
    Loads MIRACL topics+qrels (miracl/miracl) and the corresponding corpus (miracl/miracl-corpus)
    for one language (e.g., 'en', 'fr').

    Returns:
      queries_df: query_id, query
      pos_df: query_id, docid, title, text, relevance=1
      neg_df: query_id, docid, title, text, relevance=0
      corpus_df: docid, title, text
    """
    # Topics + qrels packaged as positive_passages / negative_passages per query
    miracl = load_dataset("miracl/miracl", lang)  # split: train/dev/testA depending on language :contentReference[oaicite:1]{index=1}
    ds = miracl[split]

    queries, pos_rows, neg_rows = [], [], []
    for ex in ds:
        qid = ex["query_id"]
        q = ex["query"]
        queries.append({"query_id": qid, "query": q})

        for p in ex.get("positive_passages", []):
            pos_rows.append({
                "query_id": qid,
                "docid": p["docid"],
                "title": p.get("title", ""),
                "text": p.get("text", ""),
                "relevance": 1
            })

        for n in ex.get("negative_passages", []):
            neg_rows.append({
                "query_id": qid,
                "docid": n["docid"],
                "title": n.get("title", ""),
                "text": n.get("text", ""),
                "relevance": 0
            })

        if max_rows is not None and len(queries) >= max_rows:
            break

    queries_df = pd.DataFrame(queries)
    pos_df = pd.DataFrame(pos_rows)
    neg_df = pd.DataFrame(neg_rows)

    # Corpus
    corpus = load_dataset("miracl/miracl-corpus", lang, split="train")  # docid/title/text :contentReference[oaicite:2]{index=2}
    corpus_df = corpus.to_pandas()

    return queries_df, pos_df, neg_df, corpus_df

# Example: English
q_en, pos_en, neg_en, corpus_en = load_miracl(lang="en", split="train", max_rows=200)
print(q_en.head())
print(pos_en.head())
print(corpus_en.head())

# Example: French (for your “French query breaks pipeline” scenario)
q_fr, pos_fr, neg_fr, corpus_fr = load_miracl(lang="fr", split="train", max_rows=200)
print(q_fr.head())
