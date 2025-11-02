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

### optional coding
Run DTI prediction with **DeepPurpose** in Python.

---
