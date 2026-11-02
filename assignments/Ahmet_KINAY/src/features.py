"""
Feature Extraction for Insect DNA Barcoding Sequences
Implements:
- k-mer spectrum extraction (k=3, normalized frequency distribution across 64 trinucleotides)
- GC content, AT-skew, and GC-skew
- Shannon entropy of nucleotide sequence
- Dinucleotide ratio metrics (CpG, TpA depletion)
- scikit-learn compatible transformer interface
"""

import itertools
import math
import numpy as np
import pandas as pd
from collections import Counter


class SequenceFeatureExtractor:
    """
    Extracts biological, alignment-free features from DNA sequences.
    """
    def __init__(self, k=3, include_physicochemical=True):
        self.k = k
        self.include_physicochemical = include_physicochemical
        bases = ['A', 'C', 'G', 'T']
        self.all_kmers = [''.join(p) for p in itertools.product(bases, repeat=self.k)]
        self.kmer_to_idx = {kmer: idx for idx, kmer in enumerate(self.all_kmers)}
        self.feature_names_ = []

    def _calc_kmer_freqs(self, seq):
        seq = seq.upper()
        n = len(seq)
        counts = Counter()
        for i in range(n - self.k + 1):
            kmer = seq[i:i + self.k]
            if kmer in self.kmer_to_idx:
                counts[kmer] += 1
        
        total = sum(counts.values()) or 1
        return np.array([counts[kmer] / total for kmer in self.all_kmers], dtype=np.float64)

    def _calc_physicochemical(self, seq):
        seq = seq.upper()
        n = len(seq) or 1
        a = seq.count('A')
        c = seq.count('C')
        g = seq.count('G')
        t = seq.count('T')

        gc = (g + c) / n
        at = (a + t) / n

        # Strand skews
        at_skew = (a - t) / (a + t) if (a + t) > 0 else 0.0
        gc_skew = (g - c) / (g + c) if (g + c) > 0 else 0.0

        # Shannon Entropy
        probs = [count / n for count in [a, c, g, t] if count > 0]
        entropy = -sum(p * math.log2(p) for p in probs)

        # Dinucleotide ratios (CpG and TpA observed vs expected)
        cpg_obs = seq.count("CG") / (n - 1) if n > 1 else 0
        cpg_exp = (c / n) * (g / n) if n > 0 else 1
        cpg_ratio = cpg_obs / cpg_exp if cpg_exp > 0 else 0.0

        tpa_obs = seq.count("TA") / (n - 1) if n > 1 else 0
        tpa_exp = (t / n) * (a / n) if n > 0 else 1
        tpa_ratio = tpa_obs / tpa_exp if tpa_exp > 0 else 0.0

        return [
            gc * 100.0,       # GC %
            at_skew,          # AT Skew
            gc_skew,          # GC Skew
            entropy,          # Sequence Entropy
            cpg_ratio,        # CpG observed/expected
            tpa_ratio,        # TpA observed/expected
            float(len(seq))   # Sequence length
        ]

    def fit(self, X, y=None):
        self.feature_names_ = list(self.all_kmers)
        if self.include_physicochemical:
            self.feature_names_.extend([
                "gc_percent",
                "at_skew",
                "gc_skew",
                "shannon_entropy",
                "cpg_ratio",
                "tpa_ratio",
                "seq_length"
            ])
        return self

    def transform(self, sequences):
        """
        Transforms an iterable of DNA strings into a 2D numpy array of features.
        """
        rows = []
        for seq in sequences:
            kmer_vec = self._calc_kmer_freqs(seq)
            if self.include_physicochemical:
                phys_vec = self._calc_physicochemical(seq)
                combined = np.concatenate([kmer_vec, phys_vec])
                rows.append(combined)
            else:
                rows.append(kmer_vec)
        return np.array(rows, dtype=np.float64)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self):
        return np.array(self.feature_names_)


def extract_features_df(csv_path):
    """
    Convenience function to load CSV, extract features and return as DataFrame.
    """
    df = pd.read_csv(csv_path)
    extractor = SequenceFeatureExtractor(k=3, include_physicochemical=True)
    X = extractor.fit_transform(df['sequence'])
    feature_names = extractor.get_feature_names_out()
    X_df = pd.DataFrame(X, columns=feature_names)
    X_df['order'] = df['order']
    X_df['species'] = df['species']
    X_df['sample_id'] = df['sample_id']
    return X_df, extractor


if __name__ == "__main__":
    extractor = SequenceFeatureExtractor(k=3)
    test_seq = "TCTACTAATCATAAAGATATTGGTACTTTATATTTTATTTTTGGAATTTGAGCGGGAATA"
    features = extractor.fit_transform([test_seq])
    print(f"Extracted {features.shape[1]} features from test sequence.")
    print(f"First 10 feature names: {extractor.get_feature_names_out()[:10]}")
