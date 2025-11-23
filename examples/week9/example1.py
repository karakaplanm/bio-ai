"""
Simple PubMed literature mining example

- Search PubMed for a query
- Download a limited number of abstracts
- Do a small frequency analysis on the text

Requirements:
    pip install biopython nltk --break-system-packages
    python3 -c "import nltk; nltk.download('punkt_tab')"

"""

from Bio import Entrez
from collections import Counter
import nltk

# If you run this first time:
# nltk.download("punkt")

# 1. Configure Entrez (always set your email)
Entrez.email = "your_email@example.com"

def search_pubmed(query, max_results=10):
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
    """Fetch PubMed abstracts for a list of PMIDs."""
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
    # Very simple split by double newlines (each abstract block)
    abstracts = [a.strip() for a in data.split("\n\n") if a.strip()]
    return abstracts

def preprocess_text(text):
    """Tokenize text and return lowercase word tokens."""
    tokens = nltk.word_tokenize(text)
    tokens = [t.lower() for t in tokens if t.isalpha()]
    return tokens

def main():
    # 2. Define a biomedical query
    query = "gene expression AND cancer"

    print(f"Searching PubMed for: {query}")
    pmids = search_pubmed(query, max_results=20)
    print(f"Found {len(pmids)} PMIDs")

    # 3. Fetch abstracts
    abstracts = fetch_abstracts(pmids)
    print(f"Fetched {len(abstracts)} abstract blocks")

    # 4. Simple frequency analysis
    all_tokens = []
    for abs_text in abstracts:
        tokens = preprocess_text(abs_text)
        all_tokens.extend(tokens)

    freq = Counter(all_tokens)

    # 5. Print top 30 most common words
    print("\nTop 30 most frequent words:")
    for word, count in freq.most_common(30):
        print(f"{word:<20} {count}")

if __name__ == "__main__":
    main()
