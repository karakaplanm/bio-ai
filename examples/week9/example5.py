"""
BioBERT sentence embedding + similarity example

Model:
    dmis-lab/biobert-base-cased-v1.1

Requirements:
    pip install transformers torch scikit-learn
"""

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Mean Pooling (standard method)
# ----------------------------
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


# ----------------------------
# Sentence Embedding Function
# ----------------------------
def embed(sentence, tokenizer, model):
    encoded = tokenizer(
        [sentence],
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        output = model(**encoded)

    emb = mean_pooling(output, encoded["attention_mask"])
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # normalize
    return emb.cpu().numpy()


# ----------------------------
# Main Example
# ----------------------------
def main():
    print(f"Loading BioBERT: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # Example biomedical sentences
    s1 = "BRCA1 mutations are strongly associated with breast cancer."
    s2 = "Genetic alterations in BRCA1 increase the risk of breast cancer."
    s3 = "Machine learning models perform well in image recognition tasks."

    print("\nEmbedding sentences...\n")
    e1 = embed(s1, tokenizer, model)
    e2 = embed(s2, tokenizer, model)
    e3 = embed(s3, tokenizer, model)

    # Similarity
    sim12 = cosine_similarity(e1, e2)[0][0]
    sim13 = cosine_similarity(e1, e3)[0][0]

    print(f"Similarity(s1, s2)  (both biomedical & related) : {sim12:.4f}")
    print(f"Similarity(s1, s3)  (biomedical vs general text): {sim13:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
