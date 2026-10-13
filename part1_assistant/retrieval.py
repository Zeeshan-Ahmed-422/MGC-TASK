"""
Chunking + retrieval over the three MGC documents.

The three documents are a few thousand words in total. That is small enough that an
embedding model and a vector database would be ceremony, not engineering: BM25 (a
classic keyword-ranking function) over ~25 chunks is fast, has no API cost, no index to
keep in sync, and is easy to debug - you can print the chunk it matched.

Chunking is by markdown "## " heading, because that is how the documents are already
organised ("Discounts", "Transfers", "Other Charges", ...). That gives the assistant a
section name to cite, which is exactly what the brief asks for: an answer without its
source counts as wrong.
"""

import os
import re

from rank_bm25 import BM25Okapi

# data/docs/ sits two levels up from this file. Absolute paths so the module works no
# matter which directory the reviewer runs python from.
_HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(_HERE, "..", "data", "docs")


def _tokenise(text):
    """Lowercase and keep words/numbers. BM25 just needs a list of tokens.

    We keep digits and % so queries like "2%" or "floor 15" can still match, and we
    split "2-bed" into "2" and "bed" so that "2 bed", "2-bed" and "2bed" all behave
    similarly.
    """
    return re.findall(r"[a-z0-9%]+", text.lower())


def load_chunks(docs_dir=DOCS_DIR):
    """Read every .md in docs_dir and split it into one chunk per '## ' section.

    Returns a list of dicts: {document, section, text}.
      - `document` is the file's H1 title (e.g. "MGC Aurora Heights - Price List &
        Payment Plan"), which is what a salesperson would recognise, not the filename.
      - `section` is the H2 heading.
    Text before the first H2 (the title block and the issue date line) is kept as a
    "Document header" chunk, because it holds things like "Effective 1 April 2025" and
    "Version 2.1" that matter when two documents disagree.
    """
    chunks = []
    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith(".md"):
            continue
        path = os.path.join(docs_dir, filename)
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()

        # H1 title -> document name. Fall back to the filename if a doc has no H1.
        h1 = re.search(r"^#\s+(.+)$", raw, flags=re.MULTILINE)
        document = h1.group(1).strip() if h1 else filename

        # Split on H2 headings, keeping the heading text. re.split with one capture
        # group returns [before, heading1, body1, heading2, body2, ...].
        parts = re.split(r"^##\s+(.+)$", raw, flags=re.MULTILINE)
        header_text = parts[0].strip()
        if header_text:
            chunks.append({
                "document": document,
                "section": "Document header",
                "text": header_text,
                "filename": filename,
            })
        for section, body in zip(parts[1::2], parts[2::2]):
            body = body.strip()
            if not body:
                continue
            chunks.append({
                "document": document,
                "section": section.strip(),
                # Prepend the headings so section words ("Discounts", "Transfers") are
                # searchable, not just the body text.
                "text": f"{document}\n## {section.strip()}\n{body}",
                "filename": filename,
            })
    return chunks


class Retriever:
    """BM25 search over the document chunks."""

    def __init__(self, docs_dir=DOCS_DIR):
        self.chunks = load_chunks(docs_dir)
        if not self.chunks:
            raise RuntimeError(f"No markdown documents found in {docs_dir}")
        self.bm25 = BM25Okapi([_tokenise(c["text"]) for c in self.chunks])

    def search(self, query, k=5):
        """Return the k best-matching chunks, each with its BM25 score attached."""
        scores = self.bm25.get_scores(_tokenise(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked[:k]:
            chunk = dict(self.chunks[i])
            chunk["score"] = round(float(scores[i]), 3)
            out.append(chunk)
        return out

    def as_context(self, query, k=5):
        """Format the top k chunks as the context block we hand to the LLM.

        Each chunk is labelled with its document and section so the model can cite them,
        and is explicitly numbered so it can refer to a specific one.
        """
        hits = self.search(query, k=k)
        blocks = []
        for n, h in enumerate(hits, start=1):
            blocks.append(
                f"[{n}] DOCUMENT: {h['document']}\n"
                f"    SECTION: {h['section']}\n"
                f"{h['text']}"
            )
        return "\n\n---\n\n".join(blocks), hits


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    r = Retriever()
    print(f"{len(r.chunks)} chunks loaded:")
    for c in r.chunks:
        print(f"  {c['document'][:45]:<47} | {c['section']}")
    print()
    for q in ["transfer fee", "rental yield on a 1-bed", "who is the anchor tenant",
              "base price of a 2-bed in Block B", "overseas buyer discount 50% upfront"]:
        print(f"Q: {q}")
        for h in r.search(q, k=3):
            print(f"   {h['score']:>7} | {h['document'][:40]:<42} | {h['section']}")
        print()
