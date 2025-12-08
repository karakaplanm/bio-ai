# Lecture 11: Drug Discovery with AI
*Virtual Screening, Generative Models, and Molecular Design*

## Learning Objectives
- Describe the role of AI in modern drug discovery pipelines.
- Compare structure-based vs ligand-based virtual screening approaches.
- Explain how generative models (VAEs, GANs, diffusion models, transformers) are used to design novel molecules.
- Understand molecular property prediction, ADMET estimation, and optimization workflows.
- Evaluate the advantages and limitations of AI-driven molecular design.

---

## Topics

### 1. Introduction to AI-driven Drug Discovery
- Traditional drug discovery bottlenecks  
  - High cost  
  - Long timelines (10–15 years)  
  - Low success rate  
- Where AI fits in the pipeline  
  - Target identification  
  - Hit discovery  
  - Lead optimization  
  - Preclinical prediction  

---

### 2. Virtual Screening

#### 2.1 Ligand-Based Virtual Screening (LBVS)
- Requires known actives  
- Uses similarity metrics and ML  
- Common algorithms: SVM, Random Forest, GNNs, Siamese networks  

#### 2.2 Structure-Based Virtual Screening (SBVS)
- Requires 3D target structure  
- Tools: docking, deep scoring models  
- AI methods: CNN-based scoring, pose prediction (e.g., AtomNet, GNINA)

#### 2.3 Ultra-large Scale Screening
- Screening millions–billions of compounds  
- Enabled by cloud computing and deep learning  
- Example databases: ZINC20, Enamine REAL  

---

### 3. Generative Models for Molecular Design

#### 3.1 Motivation
- Chemical space: estimated 10^60–10^100 molecules  
- AI can propose molecules with:  
  - High affinity  
  - Good ADMET  
  - Synthetic feasibility

#### 3.2 Model Types
- **VAEs**: latent space optimization  
- **GANs**: adversarial creation of new scaffolds  
- **Reinforcement Learning**: reward-driven molecular optimization  
- **Graph-based Generators**: atom-by-atom construction  
- **Diffusion Models**: state-of-the-art for stable, diverse generation  
- **Chemical Language Models** (SMILES/SELFIES transformers)

---

### 4. Molecular Property Prediction
- Key properties:  
  - Physicochemical descriptors  
  - Binding affinity  
  - Toxicity  
  - ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity)
- AI models:  
  - GNNs  
  - Transformers  
  - Multitask neural networks  

---

### 5. De Novo Design Workflow
1. Generate candidate molecules  
2. Predict properties using ML models  
3. Filter undesirable molecules (Lipinski, Veber, toxicity rules)  
4. Score or dock promising candidates  
5. Multi-objective optimization (potency, ADMET, synthesis)  
6. Select molecules for experimental validation  

---

### 6. Case Studies
- AlphaFold enabling AI-powered SBVS  
- Insilico Medicine: AI-designed molecules reaching clinical trials  
- BenevolentAI: drug repurposing via knowledge graphs  
- RL-based kinase inhibitor design  

---

## Reading

### Primary
- Walters, W. P., Barzilay, R., & Jaakkola, T. (2020). Applications of deep learning in molecule generation and discovery. *PNAS*.  
- Schneider, G. (2018). Automating drug discovery. *Science*.  
- Stokes et al. (2020). A deep learning approach to antibiotic discovery. *Cell*.  

### Additional
- Gómez-Bombarelli et al. (2018). Automatic chemical design using VAEs.  
- Jumper et al. (2021). AlphaFold protein structure prediction.  
- Brown et al. (2019). GuacaMol benchmarks for molecular design.  

---

## Notes (Narrative Summary)
AI is transforming drug discovery by significantly reducing the time and cost required to identify new therapeutic molecules. Virtual screening—both ligand-based and structure-based—uses machine learning and deep learning to rapidly assess vast chemical libraries. Generative models, including VAEs, GANs, RL systems, graph generators, and diffusion models, explore chemical space to propose novel molecules with desirable biological and pharmacokinetic properties.

By combining generation, prediction, filtering, and optimization in iterative loops, AI builds an automated molecular design pipeline. This closed-loop workflow enables researchers to move from concept to candidate molecules much faster than traditional methods allow.

---
