# Week 9: Natural Language Processing in Biology
### *Biological Literature Mining • PubMed • Language Models*

---

## 1. Introduction
Natural Language Processing (NLP) has become a critical tool in modern biology. As biological knowledge grows exponentially, automated methods are required to extract, organize, and interpret information from scientific literature, clinical notes, research databases, and online repositories.

NLP enables:
- Automated literature search
- Named entity recognition (genes, proteins, diseases)
- Relationship extraction (e.g., gene–disease associations)
- Summarization of scientific articles
- Knowledge graph construction
- Support for biomedical research and drug discovery

---

## 2. Biological Literature Mining
Biological literature mining refers to the use of computational techniques to extract structured biological information from unstructured text (papers, abstracts, reports).

### 2.1 Why Literature Mining?
- The volume of biomedical publications doubles every few years.
- Manual reading is slow, biased, and incomplete.
- Literature mining accelerates hypothesis generation.

### 2.2 Common Tasks in Literature Mining

#### • Named Entity Recognition (NER)
Identify biological entities:
- Genes (BRCA1, TP53)
- Proteins
- Diseases (COVID-19, Alzheimer’s)
- Chemicals/drugs
- Species names

Challenges:
- Ambiguous gene names ("CAT", "MAP")
- Multiple synonyms
- Domain-specific terminology

#### • Relation Extraction (RE)
Identify meaningful relationships such as:
- Gene–disease association
- Protein–protein interaction
- Drug–target interaction
- Pathways and mechanisms

Approaches:
- Pattern-based extraction
- Machine learning / deep learning
- Transformer-based models

#### • Text Classification
Used for:
- Topic categorization
- Filtering relevant literature
- Study type classification (clinical trial, review, RCT)

#### • Document Summarization
Produce concise summaries of long papers.
- Extractive vs. abstractive methods
- Transformer models improve accuracy in biomedical text

---

## 3. PubMed and Biological Databases

### 3.1 What is PubMed?
- Managed by the U.S. National Library of Medicine (NLM)
- Provides access to >36 million citations
- Focuses on life sciences, medicine, molecular biology, biochemistry

### 3.2 Key Features of PubMed
- **MeSH (Medical Subject Headings):** Controlled vocabulary system used for indexing articles.
- **PubMed ID (PMID):** Unique identifier for each article.
- **Advanced Filters:** Publication date, article type, species, language.

### 3.3 PubMed API / E-utilities
Used for automated data access:
- `esearch`: find PMIDs  
- `efetch`: download abstracts  
- `esummary`: metadata extraction  

Applications:
- Automated literature mining pipelines
- Large-scale text retrieval for machine learning

---

## 4. Language Models in Biology

### 4.1 Classical Models
- Bag-of-Words (BoW)
- TF-IDF
- n-grams  
Useful for baseline classification and keyword search but lack deep semantic understanding.

### 4.2 Word Embeddings
Represent words in continuous vector space:
- Word2Vec
- GloVe
- FastText
- BioWordVec

Advantages:
- Capture semantic similarity (“insulin” ↔ “glucose metabolism”)

### 4.3 Transformer-based Models
Transformers revolutionized biomedical NLP.

Biomedical transformer models:
- **BioBERT**
- **SciBERT**
- **PubMedBERT**
- **ClinicalBERT**
- **BioGPT**

Capabilities:
- Accurate NER and RE
- Summarization
- Question answering (QA)
- Generative scientific reasoning
- Literature-based discovery

### 4.4 Fine-tuning in Biomedical NLP
Used for:
- Specific disease domains
- Clinical terminology
- Drug development
- Knowledge extraction

---

## 5. Applications of NLP in Biology

### 5.1 Drug Discovery
- Predict drug–target relationships
- Identify adverse events from clinical reports
- Mine chemical interactions

### 5.2 Genomics and Proteomics
- Extract gene function descriptions
- Identify mutations
- Interpret protein interaction networks

### 5.3 Clinical Informatics
- Electronic health record mining
- Detect symptoms, diagnoses, treatments
- Support decision systems

### 5.4 Public Health Surveillance
- Detect outbreaks
- Analyze news and social media
- Track pandemic trends

### 5.5 Research Support for Scientists
- Summarize new papers
- Hypothesis generation
- Build knowledge graphs

---

## 6. Challenges
- Ambiguous terminology
- Lack of annotated biomedical datasets
- Sensitive/clinical data privacy
- Domain complexity
- Continuously expanding literature

---

## 7. Summary (Exam-Oriented)
- NLP enables automated extraction and interpretation of biological information.
- PubMed is the primary biomedical database with MeSH indexing and API support.
- Literature mining includes NER, relation extraction, classification, and summarization.
- Key biological language models: BioBERT, SciBERT, PubMedBERT, BioGPT.
- Applications include drug discovery, clinical informatics, genomics, and public health.

