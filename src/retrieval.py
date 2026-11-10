"""TF-IDF retrieval over kb/*.md, chunked by '## ' sections.

Chinese has no spaces, so we use character n-grams (1-3) instead of a word
tokenizer: no extra dependency, robust to typos, good enough for a tiny KB.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import KB_DIR

MIN_SCORE = 0.12  # below this we treat the KB as having no answer (tuned on a few probe queries)


@dataclass
class Passage:
    source: str
    section: str
    text: str


def load_passages(kb_dir: Path = KB_DIR) -> list[Passage]:
    passages: list[Passage] = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        content = path.read_text(encoding="utf-8")
        for block in re.split(r"\n(?=## )", content):
            block = block.strip()
            if not block.startswith("## "):
                continue  # skip title / disclaimer header
            heading, _, body = block.partition("\n")
            passages.append(Passage(path.name, heading[3:].strip(), body.strip()))
    return passages


class KBIndex:
    def __init__(self, kb_dir: Path = KB_DIR):
        self.passages = load_passages(kb_dir)
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True)
        docs = [f"{p.section} {p.section} {p.text}" for p in self.passages]  # heading weighted x2
        self.matrix = self.vectorizer.fit_transform(docs) if docs else None

    def search(self, query: str, top_k: int = 3, min_score: float = MIN_SCORE) -> list[dict]:
        if self.matrix is None or not query.strip():
            return []
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = scores.argsort()[::-1][:top_k]
        return [
            {"source": self.passages[i].source, "section": self.passages[i].section,
             "score": round(float(scores[i]), 3), "text": self.passages[i].text}
            for i in order if scores[i] >= min_score
        ]
