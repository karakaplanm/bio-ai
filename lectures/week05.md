# 🧬 Lecture 5: Genomic Data Analysis

**Course:** Computational Biology & Bioinformatics  
**Duration:** 2 hours  
**Instructor:** Dr. Mustafa Karakaplan  
**Topic Focus:** Sequence alignment, classification, and prediction tasks in genomic data analysis  

---

## 1. Introduction to Genomic Data

### 1.1. What is Genomic Data?
Genomic data refers to the **complete set of DNA sequences** within an organism, including all genes and non-coding regions.  
It enables the understanding of biological function, variation, and disease mechanisms.

### 1.2. Major Data Types

| Type | Description | Common File Formats |
|------|--------------|---------------------|
| **DNA Sequence** | Nucleotides (A, T, G, C) | `.fasta`, `.fastq` |
| **RNA-seq** | Gene expression data | `.bam`, `.sam`, `.gtf` |
| **Variants** | SNPs, INDELs, CNVs | `.vcf`, `.bcf` |
| **Annotations** | Gene structure, function | `.gff3`, `.bed` |

### 1.3. Sources of Genomic Data
- **NCBI GenBank:** Reference genome sequences  
- **Ensembl:** Annotated genome databases  
- **1000 Genomes Project:** Human population variation data  
- **GEO (Gene Expression Omnibus):** Transcriptomic datasets  

---

## 2. Sequence Alignment

Sequence alignment is the process of arranging sequences to identify **regions of similarity** that may indicate functional, structural, or evolutionary relationships.

### 2.1. Types of Alignment

| Type | Description | Example Algorithms |
|------|--------------|--------------------|
| **Global Alignment** | Aligns entire sequences end-to-end | Needleman–Wunsch |
| **Local Alignment** | Finds most similar subregions | Smith–Waterman |
| **Multiple Sequence Alignment (MSA)** | Aligns more than two sequences simultaneously | ClustalW, MUSCLE, MAFFT |

### 2.2. Scoring System
- **Match:** +1  
- **Mismatch:** -1  
- **Gap penalty:** -2  
- **Substitution matrices:**  
  - *PAM* (Point Accepted Mutation)  
  - *BLOSUM* (Blocks Substitution Matrix)

### 2.3. Example: Biopython Alignment
```python
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

seq1 = "ATGCT"
seq2 = "ATGTT"

alignments = pairwise2.align.globalxx(seq1, seq2)
for a in alignments:
    print(format_alignment(*a))
```

**Output Example:**
```
ATGCT
||| |
ATGTT
Score=4
```

### 2.4. Tools
- **BLAST (Basic Local Alignment Search Tool):**  
  Compares a query sequence against a database.
- **Clustal Omega:**  
  Efficient multiple sequence alignment.
- **MAFFT:**  
  Fast MSA for large genomic datasets.

---

## 3. Classification in Genomics

Classification tasks in genomics often aim to **categorize biological samples** or **predict gene functions** using computational models.

### 3.1. Feature Extraction from Sequences
- **k-mer Analysis:** Count of subsequences of length *k*.  
  Example: For `ATGC`, 2-mers are `AT`, `TG`, `GC`.
- **GC Content:**  
  \[
  GC\% = \frac{G + C}{A + T + G + C} \times 100
  \]
- **Motif Detection:** Searching conserved short patterns in DNA or proteins.
- **One-hot Encoding:** Representing each nucleotide as a binary vector.  
  Example:  
  A = [1,0,0,0], T = [0,1,0,0], G = [0,0,1,0], C = [0,0,0,1]

### 3.2. Machine Learning Models

| Model | Use Case | Notes |
|--------|-----------|-------|
| **Logistic Regression** | Binary gene classification | Fast and interpretable |
| **Random Forest** | Gene expression classification | Handles nonlinear features |
| **SVM** | Promoter vs. non-promoter | Works well with high-dimensional data |
| **Neural Networks (CNN/RNN)** | Sequence-based prediction | Captures local and long-range dependencies |

### 3.3. Example: Gene Expression Classification
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("gene_expression.csv")
X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))
```

---

## 4. Prediction Tasks in Genomic Analysis

Prediction tasks use existing data to **infer unknown biological attributes** such as gene function, regulatory regions, or protein structure.

### 4.1. Gene Function Prediction
- **Homology-based prediction:** Using known gene homologs.  
- **Machine Learning-based:** Predict Gene Ontology (GO) terms.  
- **Deep Learning:** Feature learning directly from sequence input.

### 4.2. Promoter and Enhancer Region Detection
- DNA sequences have specific motifs (e.g., TATA box) indicating regulatory roles.  
- CNN-based architectures are successful at discovering motif patterns.

**CNN-based motif prediction framework:**
```
Input DNA sequence → Convolution layers → Pooling → Dense → Output (Promoter / Non-Promoter)
```

### 4.3. Protein Structure Prediction
- Sequence → Secondary / tertiary structure prediction.  
- **AlphaFold** revolutionized the field using deep learning on sequence co-evolution data.  

---

## 5. Tools and Resources

| Tool | Description |
|------|--------------|
| **Biopython** | Python library for sequence analysis |
| **BLAST** | Local sequence similarity search |
| **Clustal Omega / MAFFT** | Multiple sequence alignment |
| **scikit-bio** | Bioinformatics analysis in Python |
| **TensorFlow / PyTorch** | Deep learning for genomics |
| **GEO / TCGA** | Gene expression and cancer genomics data sources |

---

## 6. Case Study: Predicting Gene Function from Sequence

### Dataset
- 10,000 gene sequences labeled with known functions.  
- Extracted 6-mer frequency features.

### Model
- Random Forest classifier  
- Accuracy: **87%**  
- Most informative features: GC content and AT-rich motifs.

### Takeaway
Feature engineering in genomics is crucial; ML success depends on biologically meaningful representations.

---

## 7. Key Challenges in Genomic Data Analysis
- **Data Volume:** Terabytes of raw sequencing data.  
- **Noise:** Experimental and biological variability.  
- **Interpretability:** Models must yield biological insights.  
- **Integration:** Combining genomics, transcriptomics, and proteomics data.

---

## 8. Further Reading

1. Mount, D. W. (2004). *Bioinformatics: Sequence and Genome Analysis.* Cold Spring Harbor Laboratory Press.  
2. Altschul, S. F. et al. (1990). *Basic Local Alignment Search Tool.* J. Mol. Biol.  
3. Angermueller, C. et al. (2016). *Deep Learning for Computational Biology.* Mol. Syst. Biol.  
4. Larranaga, P. et al. (2006). *Machine Learning in Bioinformatics.* Brief. Bioinform.

---

## 9. Summary Notes

- Genomic data analysis integrates **biological knowledge and computational methods**.  
- Sequence alignment identifies similarity and evolutionary relationships.  
- Classification and prediction tasks enable functional genomics and disease research.  
- The future lies in **multi-omics integration** and **explainable deep learning** in biology.

---

## 10. Practice Exercises

1. **Compute GC Content:**  
   Write a Python function to calculate GC% for any given DNA sequence.  
2. **BLAST Search:**  
   Use NCBI BLAST to find homologous sequences for a given gene.  
3. **k-mer Frequency Analysis:**  
   Implement 3-mer feature extraction using Python.  
4. **Classification Task:**  
   Train an SVM to classify promoter vs. non-promoter sequences.  
5. **Mini Project:**  
   Choose a small dataset (e.g., 100 sequences) and attempt to predict gene function using machine learning.

---

**End of Lecture 5**

