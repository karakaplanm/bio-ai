# Lecture 6: Proteomics and AI

**Topic:** Protein Structure, AlphaFold, Drug–Target Interaction  

---

## Learning Objectives

- Understand the scope of proteomics and how it differs from genomics  
- Explain protein structure levels and functional relevance  
- Describe the fundamentals of AlphaFold and deep learning–based structure prediction  
- Discuss AI techniques for drug–target interaction (DTI) prediction  
- Evaluate the impact and limitations of AI in biomedical research  

---

## 1. Introduction to Proteomics (15 min)

### What is Proteomics?
Proteomics = large-scale study of proteins, including:
- Structure  
- Function  
- Expression  
- Interactions  

> The proteome is **dynamic**, unlike the **static genome**.

### Why Proteomics Matters
- Disease mechanisms  
- Biomarker discovery  
- Drug target identification  
- Personalized medicine  

### Key Experimental Techniques
| Method | Use |
|---|---|
| Mass Spectrometry (MS) | Protein ID & quantification |
| 2D Gel Electrophoresis | Protein separation |
| Western Blot | Protein detection |
| NMR | Structural analysis |
| Cryo-EM | High-resolution structure |

---

## 2. Protein Structure Fundamentals (15 min)

| Level | Description |
|---|---|
| Primary | Amino acid sequence |
| Secondary | α-helix, β-sheet |
| Tertiary | 3D folding |
| Quaternary | Multi-subunit complexes |

**Misfolding & Disease:** Alzheimer’s, Parkinson’s, prion disorders  

**Drug design relevance:** Active sites & binding pockets depend on structure  

---

## 3. AlphaFold and AI-based Structural Biology (40 min)

### Traditional Structure Determination
| Method | Limitations |
|---|---|
| X-ray | Slow, crystallization required |
| NMR | Small molecules only |
| Cryo-EM | Expensive |

### AlphaFold Overview
- Deep learning + attention models  
- Inputs: sequence + MSA  
- Output: 3D structure + pLDDT confidence score  

### Impact
- Millions of protein structures predicted  
- Accelerated biology & drug discovery  
- Open global database

### Limitations
- Dynamic conformations  
- Protein–protein complexes (improved in AF-Multimer)  
- Ligand binding prediction limited  

---

## 4. Drug–Target Interaction (DTI) & AI (35 min)

### What is DTI?
Interaction between a drug molecule and its protein target.

### Traditional Method
- High-throughput screening (HTS) → expensive & slow

### AI Approaches
| Method | Description |
|---|---|
| ML (SVM, RF) | Feature-based |
| Deep Learning | End-to-end learning |
| Graph Neural Networks | Molecules as graphs |
| Hybrid Models | Docking + AI |

### Common Tools
- AlphaFold + virtual screening  
- DeepPurpose  
- GraphDTA  
- DeepDock  

### Applications
- Cancer therapy  
- Rare diseases  
- Drug repurposing  
- Personalized medicine  

---

## 5. Ethics & Challenges (10 min)

- Data bias in biological datasets  
- Model interpretability concerns  
- Clinical validation still required  
- Data privacy & responsible research  

---

## Discussion Questions

1. Why is proteomics more complex than genomics for AI?
2. What are the main limitations of AlphaFold?
3. Can AI fully automate drug discovery?

---

## Recommended Reading

- Jumper et al., *Science*, 2021 — AlphaFold  
- Varadi et al., *NAR*, 2022 — AlphaFold Database  
- Gaudelet et al., *Nat. Mach. Intell.*, 2021 — GNNs for drug discovery  
- **Lodish**, *Molecular Cell Biology* — Proteomics chapter  

---

## Assignment

### mini-task
Search a protein in AlphaFold DB and report:
- Structure preview screenshot
- pLDDT confidence score
- Known or predicted function


## Example

Run DTI prediction with **DeepPurpose** in Python.

---

This workflow demonstrates how modern bioinformatics and AI tools help us analyze proteins and predict drug–target interactions efficiently.

---

### 1) Download Protein Sequence from UniProt

**What we do:**  
Retrieve the amino acid sequence of a target protein from the UniProt database.

**Why it's important:**  
The protein sequence is the fundamental input for all further analysis.  
It allows us to:

- Predict structure
- Study function
- Investigate as a drug target

> **Protein sequence = the identity of the protein**

---

### 2) Download AlphaFold Model

**What we do:**  
Use AlphaFold to obtain the predicted **3D structure** of the protein.

**Why it's important:**  
Protein **shape determines function**.

This helps us to:

- Identify active/binding sites
- Study protein mechanics
- Support rational drug design

> Without structure, drug discovery is like **shooting in the dark**.

---

### 3) Analyze pLDDT Confidence Scores

**What we do:**  
Examine pLDDT values (0–100) in the AlphaFold model.

| Score | Meaning |
|---|---|
| 90+ | Very high confidence |
| 70–90 | Good prediction |
| 50–70 | Low confidence |
| < 50 | Uncertain / unstructured region |

**Why it's important:**  
Shows which regions of the model are **reliable** and where caution is needed.

> Think of it as checking the **accuracy rating** of the model.

---

### 4) Predict Drug–Target Binding with DeepPurpose

**What we do:**  
Feed a **drug SMILES string** and the **protein sequence** into a pretrained DeepPurpose model to predict binding affinity.

**Why it's important:**

- Quickly evaluate if a drug can bind a protein
- Prioritize compounds before wet-lab experiments
- Speed up drug discovery and drug repurposing

**Interpretation:**

- High score → likely good binding
- Low score → weak/no binding

> AI gives us a **virtual screening shortcut** before laboratory testing.

---

### 🎯 Summary

| Step | Goal |
|---|---|
Get protein sequence | Identify the target |
Get AlphaFold structure | Understand 3D shape |
Check pLDDT | Validate prediction confidence |
Run DeepPurpose | Predict drug binding |

**Outcome:**  
AI + bioinformatics allows us to explore proteins and drug interactions **in minutes instead of months**.

---

### 🧠 Big Idea

Combining:

- **Structural biology**
- **Machine learning**
- **Drug discovery tools**

gives us a powerful pipeline for modern biomedical research.

---


