"""
BioBERT NER example (DISEASE + CHEMICAL)
Working model: kamalkraj/bio-ner-bert

Requirements:
    pip install transformers torch
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_NAME = "dmis-lab/biobert-v1.1"


def main():
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

    nlp_ner = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )

    text = (
        "Mutations in the BRCA1 gene increase the risk of breast cancer. "
        "Patients are often treated with tamoxifen or paclitaxel."
    )

    print("\nInput text:")
    print(text)
    print("\nEntities:\n")

    entities = nlp_ner(text)

    for ent in entities:
        print(f"{ent['word']:<20} {ent['entity_group']:<10} score={ent['score']:.3f}")


if __name__ == "__main__":
    main()
