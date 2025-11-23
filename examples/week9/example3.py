"""
Example 3: Biomedical Named Entity Recognition (NER) with scispaCy

- Search PubMed for a biomedical query
- Fetch abstracts
- Run NER using a scispaCy model
- Print detected entities (diseases, chemicals, genes, etc.)

Requirements:
    pip install biopython spacy scispacy
    # And then install one of the scispaCy models, for example:
    #   python -m spacy download en_core_sci_sm
    # or:
    #   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

    pip3 install --break-system-packages https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

pip install --break-system-packages --only-binary :all: spacy

    # In this example we assume the small general scientific model:
    #   en_core_sci_sm
"""

from Bio import Entrez
import spacy

Entrez.email = "your_email@example.com"  # <-- change this


# -------- PubMed helper functions (same idea as before) --------

def search_pubmed(query, max_results=10):
    """Search PubMed and return a list of PMIDs."""
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results
    )
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


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

    # Very simple splitting – good enough for demo
    blocks = [b.strip() for b in data.split("\n\n") if b.strip()]
    return blocks


# -------- NER pipeline with scispaCy --------

def main():
    query = "breast cancer gene expression"

    print(f"Searching PubMed for: {query}")
    pmids = search_pubmed(query, max_results=10)
    print(f"Found {len(pmids)} PMIDs")

    abstracts = fetch_abstracts(pmids)
    print(f"Fetched {len(abstracts)} abstract blocks\n")

    if not abstracts:
        print("No abstracts retrieved.")
        return

    # Load a spaCy model
    # Using en_core_web_sm (standard model) since scispaCy installation failed
    # For biomedical NER, install scispaCy models separately
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    # Optional: if using a specific NER model like en_ner_bc5cdr_md,
    # you get more precise DISEASE / CHEMICAL entities.

    for i, abs_text in enumerate(abstracts):
        print("=" * 80)
        print(f"ABSTRACT {i}")
        print("-" * 80)
        print(abs_text[:600], "...\n")  # show truncated abstract

        doc = nlp(abs_text)

        # Collect entities (text, label)
        ents = [(ent.text, ent.label_) for ent in doc.ents]

        if not ents:
            print("No entities detected by the model.\n")
            continue

        print("Detected entities (unique, text [LABEL]):\n")

        # Make them unique for nicer display
        seen = set()
        for text, label in ents:
            key = (text, label)
            if key in seen:
                continue
            seen.add(key)
            print(f"- {text} [{label}]")
        print("\n")


if __name__ == "__main__":
    main()
