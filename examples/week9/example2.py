"""
Example 2: PubMed abstracts + TF-IDF + similarity search

- Search PubMed for a biomedical query
- Download abstracts
- Build TF-IDF vectors
- Find the most similar abstracts to a chosen one

Requirements:
    pip install biopython scikit-learn nltk
"""

from Bio import Entrez
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# If you run this for the first time:
# nltk.download("punkt")

Entrez.email = "your_email@example.com"  # <-- change this


# ---------- PubMed helper functions ----------

def search_pubmed(query, max_results=20):
    """Search PubMed and return a list of PMIDs."""
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results
    )
    record = Entrez.read(handle)
    handle.close()
    pmids = record["IdList"]
    return pmids


def fetch_abstracts(pmids):
    """Fetch PubMed abstracts (plain text) for a list of PMIDs."""
    if not pmids:
        return []

    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(pmids),
        rettype="abstract",
        retmode="text"
    )
    data = handle.read()
    handle.close()

    # This is a very simple split; for demos it is enough.
    # Each abstract block is separated by one or more blank lines.
    blocks = [b.strip() for b in data.split("\n\n") if b.strip()]
    return blocks


# ---------- Preprocessing (optional) ----------

def nltk_tokenizer(text):
    """
    Simple tokenizer wrapper for TF-IDF, using NLTK.
    Converts to lowercase and keeps only alphabetic tokens.
    """
    tokens = nltk.word_tokenize(text)
    tokens = [t.lower() for t in tokens if t.isalpha()]
    return tokens


# ---------- Main TF-IDF + similarity pipeline ----------

def main():
    query = "gene expression AND cancer"

    print(f"Searching PubMed for: {query}")
    pmids = search_pubmed(query, max_results=30)
    print(f"Found {len(pmids)} PMIDs")

    abstracts = fetch_abstracts(pmids)
    print(f"Fetched {len(abstracts)} abstract blocks")

    if len(abstracts) < 2:
        print("Not enough abstracts for similarity analysis.")
        return

    # Show first few characters of each abstract (for orientation)
    print("\nSample abstracts (truncated):")
    for i, abs_text in enumerate(abstracts[:3]):
        print(f"\n--- Abstract {i} ---")
        print(abs_text[:400], "...")
    
    # Choose one abstract as the "query document"
    # Here we take the first abstract (index 0)
    query_index = 0
    query_abstract = abstracts[query_index]
    print(f"\nUsing abstract {query_index} as reference for similarity search.\n")

    # Build TF-IDF matrix for all abstracts
    vectorizer = TfidfVectorizer(
        tokenizer=nltk_tokenizer,    # use our custom tokenizer
        stop_words="english",        # remove English stopwords
        max_features=5000            # limit vocabulary size (for speed)
    )

    tfidf_matrix = vectorizer.fit_transform(abstracts)

    # Compute cosine similarity between the chosen abstract
    # and all others in the corpus
    similarities = cosine_similarity(
        tfidf_matrix[query_index:query_index+1],  # shape (1, n_features)
        tfidf_matrix                              # shape (n_docs, n_features)
    ).flatten()

    # Sort indices by similarity (highest first)
    ranked_indices = similarities.argsort()[::-1]

    print("Most similar abstracts to reference:\n")
    for rank, idx in enumerate(ranked_indices[:10]):
        # Skip itself if you want
        if idx == query_index:
            continue

        print(f"Rank {rank} | Abstract index: {idx} | Similarity: {similarities[idx]:.3f}")
        print(abstracts[idx][:400], "...\n")

    # Optionally, show the TF-IDF vocabulary size
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")


if __name__ == "__main__":
    main()
