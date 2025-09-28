## Lecture 2: Machine Learning Fundamentals  

---

## 1. Supervised vs. Unsupervised Learning  

### Supervised Learning  
- **Definition:** Model is trained with labeled data.  
- **Goal:** Learn the mapping from input → output.  
- **Examples (biology):**  
  - Disease diagnosis (images labeled as healthy/diseased)  
  - Predicting cancer type from gene expression levels  
  - Classifying proteins based on sequence into functional categories  

**Advantages:**  
- High accuracy (if sufficient labeled data).  
- Models like decision trees or linear regression can be interpretable.  

**Challenges:**  
- Requires large amounts of labeled biological data.  
- Labeling is expensive and time-consuming.  

---

### Unsupervised Learning  
- **Definition:** Model works on unlabeled data to discover patterns or structures.  
- **Goal:** Find hidden structures in the data.  
- **Examples (biology):**  
  - Clustering gene expression profiles (e.g., cancer subtypes)  
  - Identifying cell types in single-cell RNA-seq  
  - Grouping microbial communities in metagenomics  

**Advantages:**  
- Does not require labeled data.  
- Useful for discovering new biological insights.  

**Challenges:**  
- Interpretation can be difficult.  
- Sensitive to algorithm parameters (e.g., number of clusters).  

---

## 2. Biological Datasets  

### Common Types of Biological Data  
- **Genomic data** (DNA sequences)  
- **Transcriptomic data** (RNA-seq, gene expression)  
- **Proteomic data** (protein sequences, mass spectrometry)  
- **Metabolomic data** (metabolites, chemical profiles)  
- **Medical imaging** (MRI, CT, histopathology)  
- **Electronic Health Records (EHRs)**  

### Key Characteristics of Biological Data  
- **High dimensionality:**  
  - Example: 20,000+ genes but few samples.  
- **Noise:**  
  - Measurement errors, biological variability.  
- **Missing values:**  
  - Common in clinical trials.  
- **Limited sample size:**  
  - Especially for rare diseases.  

---

## 3. Topics Covered  
- Differences between supervised and unsupervised learning  
- Structure of biological datasets  
- Applications of ML in biology:  
  - Classification (disease prediction)  
  - Clustering (cell types, microbial communities)  
  - Dimensionality reduction (PCA, t-SNE, UMAP)  
  - Feature selection (biomarker discovery)  

---

## 4. Reading  
- Hastie, Tibshirani, Friedman: *The Elements of Statistical Learning* (Chapters 2–3)  
- Larrañaga et al., *Machine learning in bioinformatics* (Briefings in Bioinformatics, 2006)  
- Angermueller et al., *Deep learning for computational biology* (Molecular Systems Biology, 2016)  

---

## 5. Notes (Summary)  
- **Supervised learning**: labeled → prediction.  
- **Unsupervised learning**: unlabeled → structure discovery.  
- **Biological data**: typically high-dimensional, noisy, incomplete, and limited in sample size.  
- **Machine learning** plays a central role in both biological discovery and clinical applications.  

---
