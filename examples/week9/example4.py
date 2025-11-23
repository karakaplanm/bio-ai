"""
Example 4: Sentence embeddings + similarity with PubMedBERT (or BioBERT)

- Take a list of biomedical sentences (e.g. from abstracts)
- Compute embeddings using a transformer model
- Compute cosine similarity between sentences

Requirements:
    pip install transformers torch scikit-learn

Optionally, you can switch the model to a BioBERT variant (see below).
"""

from typing import List

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


# ------------------ MODEL CONFIG ------------------

# PubMedBERT (biomedical, pre-trained on PubMed)
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

# If you prefer BioBERT instead, comment the above and uncomment one of these:
# MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
# MODEL_NAME = "nboost/pt-biobert-base-msmarco"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------ EMBEDDING FUNCTION ------------------

def mean_pooling(model_output, attention_mask):
    """
    Mean pooling: average token embeddings, ignoring padding tokens.
    """
    token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_dim)
    # Expand attention mask to match token_embeddings shape
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # Sum embeddings and divide by number of valid tokens
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def embed_sentences(
    sentences: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    max_length: int = 256
) -> torch.Tensor:
    """
    Compute sentence embeddings for a list of sentences.
    Returns a tensor of shape (n_sentences, hidden_dim).
    """
    encoded_input = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    # Normalize embeddings (optional but useful for cosine similarity)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu()


# ------------------ MAIN DEMO ------------------

def main():
    print(f"Using model: {MODEL_NAME}")
    print(f"Device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    # Example biomedical sentences (you can replace with sentences from PubMed)
    sentences = [
        "Gene expression profiling reveals novel biomarkers for breast cancer.",
        "Mutations in the BRCA1 gene are associated with an increased risk of breast cancer.",
        "Deep learning models are widely used for image classification in computer vision.",
        "Targeted therapies have improved survival rates in patients with lung cancer.",
        "Electronic health records can be mined to identify adverse drug reactions."
    ]

    print("Input sentences:\n")
    for i, s in enumerate(sentences):
        print(f"[{i}] {s}")
    print("\nComputing embeddings...\n")

    embeddings = embed_sentences(sentences, tokenizer, model)

    # Convert embeddings to numpy for sklearn
    emb_np = embeddings.numpy()
    sim_matrix = cosine_similarity(emb_np, emb_np)

    print("Cosine similarity matrix (rows/cols = sentences):\n")
    # Pretty print
    for i in range(len(sentences)):
        row = " ".join(f"{sim_matrix[i, j]:.3f}" for j in range(len(sentences)))
        print(f"Sentence {i}: {row}")

    # Additionally: show the most similar sentence for each (excluding itself)
    print("\nMost similar sentence to each one (excluding itself):\n")
    for i in range(len(sentences)):
        sims = sim_matrix[i].copy()
        sims[i] = -1.0  # ignore self
        j = sims.argmax()
        print(f"- Sentence {i} is most similar to sentence {j} (similarity = {sim_matrix[i, j]:.3f})")


if __name__ == "__main__":
    main()
