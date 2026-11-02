"""
Inference & Prediction Module for Insect DNA Barcoding Sequences
Loads the saved model bundle (models/best_insect_classifier.joblib) and predicts
the insect taxonomic order for any given DNA sequence.

Usage examples:
    # 1. Run demo prediction on built-in test sequences:
    python src/predict.py

    # 2. Predict for a custom raw sequence:
    python src/predict.py --sequence "TCTACTAATCATAAAGATATTGGTACTTTATATTTTATTTTTGGAATTTGAGCGGGAATA..."

    # 3. Predict from FASTA file:
    python src/predict.py --fasta data/insect_coi_sequences.fasta --index 5
"""

import os
import sys
import argparse
import joblib
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def load_model(model_path=None):
    if model_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "..", "models", "best_insect_classifier.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train.py first!")
    
    bundle = joblib.load(model_path)
    return bundle


def predict_sequence(sequence, bundle):
    """
    Predicts taxonomic order and calculates biological properties for a DNA sequence.
    """
    clean_seq = "".join([c.upper() for c in sequence if c.upper() in "ACGT"])
    if len(clean_seq) < 30:
        raise ValueError(f"Sequence is too short ({len(clean_seq)} bp). Minimum 30 bp required.")

    extractor = bundle['extractor']
    pipeline = bundle['pipeline']
    label_encoder = bundle['label_encoder']
    classes = bundle['classes']

    # Extract features
    feat_vec = extractor.transform([clean_seq])

    # Predict class
    pred_idx = pipeline.predict(feat_vec)[0]
    pred_order = label_encoder.inverse_transform([pred_idx])[0]

    # Probabilities / Decision function
    clf = pipeline.named_steps['classifier']
    if hasattr(clf, "predict_proba"):
        scaler = pipeline.named_steps['scaler']
        scaled_x = scaler.transform(feat_vec)
        probs = clf.predict_proba(scaled_x)[0]
    elif hasattr(clf, "decision_function"):
        scaler = pipeline.named_steps['scaler']
        scaled_x = scaler.transform(feat_vec)
        scores = clf.decision_function(scaled_x)[0]
        # Softmax over decision scores
        exp_s = np.exp(scores - np.max(scores))
        probs = exp_s / exp_s.sum()
    else:
        probs = np.zeros(len(classes))
        probs[pred_idx] = 1.0

    # Biological metrics
    gc = (clean_seq.count('G') + clean_seq.count('C')) / len(clean_seq) * 100.0
    a = clean_seq.count('A')
    t = clean_seq.count('T')
    at_skew = (a - t) / (a + t) if (a + t) > 0 else 0.0

    prob_dict = {cls_name: round(float(p) * 100, 2) for cls_name, p in zip(classes, probs)}
    prob_dict = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

    return {
        "predicted_order": pred_order,
        "confidence": prob_dict[pred_order],
        "all_probabilities": prob_dict,
        "sequence_length": len(clean_seq),
        "gc_percent": round(gc, 2),
        "at_skew": round(at_skew, 4)
    }


def parse_fasta(fasta_path):
    headers = []
    seqs = []
    current_hdr = None
    current_seq = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_hdr is not None:
                    headers.append(current_hdr)
                    seqs.append("".join(current_seq))
                current_hdr = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_hdr is not None:
            headers.append(current_hdr)
            seqs.append("".join(current_seq))
    return headers, seqs


def main():
    parser = argparse.ArgumentParser(description="Predict Insect Taxonomic Order from DNA Sequence")
    parser.add_argument("--sequence", type=str, default=None, help="Raw DNA sequence string")
    parser.add_argument("--fasta", type=str, default=None, help="Path to FASTA file")
    parser.add_argument("--index", type=int, default=0, help="Sequence index in FASTA file (default: 0)")
    parser.add_argument("--model", type=str, default=None, help="Path to model bundle")
    args = parser.parse_args()

    print("=" * 65)
    print("  INSECT PHYLOGENETICS & TAXONOMIC AI CLASSIFIER - INFERENCE  ")
    print("=" * 65)

    bundle = load_model(args.model)
    print(f"Loaded Trained Model: {bundle['model_name']}")

    seq_to_test = None
    header_info = "Manual Input"

    if args.sequence:
        seq_to_test = args.sequence
    elif args.fasta:
        headers, seqs = parse_fasta(args.fasta)
        if args.index >= len(seqs):
            print(f"Error: FASTA contains only {len(seqs)} sequences (index {args.index} out of range).")
            return
        header_info = headers[args.index]
        seq_to_test = seqs[args.index]
    else:
        # Demo sequence (Western Honeybee / Apis mellifera sample)
        print("\n[*] No input specified. Running test demonstration on representative sample...")
        fasta_default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "insect_coi_sequences.fasta")
        if os.path.exists(fasta_default):
            headers, seqs = parse_fasta(fasta_default)
            # Pick sample #155 (Hymenoptera)
            sample_idx = min(155, len(seqs) - 1)
            header_info = headers[sample_idx]
            seq_to_test = seqs[sample_idx]
        else:
            header_info = "Synthetic Apis mellifera (Hymenoptera)"
            seq_to_test = "TCTACTAATCATAAAGATATTGGTACTTTATATTTTATTTTTGGAATTTGAGCGGGAATA" * 10

    result = predict_sequence(seq_to_test, bundle)

    print("\n---------------- SEQUENCE METRICS ----------------")
    print(f"Header / Source  : {header_info}")
    print(f"Sequence Length  : {result['sequence_length']} bp")
    print(f"GC Content       : {result['gc_percent']}%")
    print(f"Strand AT-Skew   : {result['at_skew']}")
    print("--------------------------------------------------")
    print(f"\n[★] PREDICTED INSECT ORDER: {result['predicted_order']}")
    print(f"[★] MODEL CONFIDENCE      : {result['confidence']}%")

    print("\nTaxonomic Class Probabilities:")
    for order, prob in result['all_probabilities'].items():
        bar = "█" * int(prob / 4)
        print(f"  {order:14s} : {prob:6.2f}%  {bar}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
