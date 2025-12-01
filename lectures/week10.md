# Lecture 10: AI for Systems Biology

### Networks, Pathways, Multi-Omics Data Integration

---

## 1. Introduction: Systems Biology and AI
Systems biology interprets cellular and organism-level processes as **integrated networks**, not isolated components. Instead of focusing on single genes or proteins, it analyzes dynamic interactions across **genes, proteins, metabolites**, and regulatory mechanisms.

AI (especially ML and DL) helps by:
- uncovering hidden patterns  
- predicting disease mechanisms  
- identifying biomarkers  
- discovering drug targets  

---

## 2. Biological Networks

Biological networks consist of **nodes** (genes, proteins, metabolites) and **edges** (interactions, regulation).

### 2.1 Types of Networks
- Protein–Protein Interaction (PPI) Networks  
- Gene Regulatory Networks (GRN)  
- Metabolic Networks  
- Co-expression Networks  
- Signaling Networks  

### 2.2 Network Properties
- **Degree** – number of edges per node  
- **Hub nodes** – highly connected and often essential  
- **Betweenness centrality** – controls information flow  
- **Community detection** – functional modules  

### 2.3 AI Methods for Network Analysis
- Graph Neural Networks (GNNs)  
- Node embedding (Node2Vec, DeepWalk)  
- Link prediction and network completion  

---

## 3. Biological Pathways

Pathways represent structured sequences of biochemical events and regulatory steps.

### Examples
- MAPK/ERK  
- PI3K–AKT–mTOR  
- Apoptosis pathway  
- p53 DNA damage response  

### AI Applications in Pathway Analysis
- pathway activation prediction  
- disease subtype classification  
- therapy response prediction  

---

## 4. Multi-Omics Data and Integration

Modern biology uses diverse omics layers:
- Genomics  
- Epigenomics  
- Transcriptomics  
- Proteomics  
- Metabolomics  

Integration is necessary to capture the full system.

### 4.1 Integration Strategies
- **Early Integration** – concatenate features  
- **Intermediate Integration** – latent shared embedding  
- **Late Integration** – combine decisions after separate analyses  

### 4.2 AI Techniques
- Autoencoders (including VAE)  
- Multi-modal deep learning  
- Graph neural networks  
- Clustering (k-means, spectral methods)  
- Bayesian models  

---

## 5. Applications

### 5.1 Disease Subtype Discovery
Multi-omics analysis enables detection of new disease subtypes (e.g., cancer).

### 5.2 Biomarker Discovery
AI identifies:
- critical network nodes  
- pathway activity patterns  
- prognostic markers  

### 5.3 Drug Target Identification & Repurposing
Knowledge graphs + GNNs enable:
- new drug–target predictions  
- combination therapy suggestions  

### 5.4 Single-Cell Analysis
AI supports:
- cell-type classification  
- trajectory inference  
- cell–cell communication modeling  

---

## 6. AI-Based Interpretation of Biological Networks

### 6.1 Graph Neural Networks (GNNs)
GNNs model complex, heterogeneous systems:
- multiple node types  
- multiple interaction types  

### 6.2 Node Embeddings
Generate vector representations for biological entities.

Common methods:
- Node2Vec  
- DeepWalk  
- LINE  
- GraphSAGE  

---

## 7. Data Preprocessing

Before modeling:
- apply batch-effect correction  
- normalize data  
- impute missing values  
- perform PCA or UMAP  

---

## 8. Limitations & Ethics

### Limitations
- heterogeneous data  
- noise in experiments  
- limited interpretability  

### Ethical Issues
- genomic data privacy  
- transparency in clinical systems  
- explainability requirements  

---

## 9. Summary

Key points:
- networks and pathways form the basis of systems biology  
- AI helps analyze complex interactions  
- multi-omics integration enables disease stratification, biomarker discovery, drug development  
- AI is essential for modern precision medicine  

