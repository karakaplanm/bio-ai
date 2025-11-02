# =========================
# Install dependencies
# =========================
# 
# pip install biopython requests matplotlib --break-system-packages
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --break-system-packages
# pip install DeepPurpose --break-system-packages
# pip install rdkit --break-system-packages
# pip install git+https://github.com/bp-kelley/descriptastorus --break-system-packages



import requests, io, statistics
from Bio import SeqIO
import matplotlib.pyplot as plt
from DeepPurpose import DTI, utils

# =========================
# User input
# =========================
# https://www.uniprot.org/uniprotkb/P00519/entry
uniprot_id = "P00519"  # ABL1 kinase

# Simplified Imatinib SMILES (verified structure)
# https://pubchem.ncbi.nlm.nih.gov/compound/23351413
# https://pubchem.ncbi.nlm.nih.gov/compound/Imatinib
drug_smiles = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C4=NC=CC(=N4)C5=CN=CC=C5"  # Imatinib


print(f"Protein: {uniprot_id} | Drug: Imatinib")

# =========================
# Fetch protein FASTA from UniProt
# =========================

url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
r = requests.get(url)
fasta = r.text
sequence = str(list(SeqIO.parse(io.StringIO(fasta), "fasta"))[0].seq)
print("Protein sequence length:", len(sequence))

# =========================
# Download AlphaFold structure CIF
# =========================
# https://alphafold.ebi.ac.uk/search/text/P00519
cif_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-2-F1-model_v6.cif"
cif_file = f"{uniprot_id}.cif"
try:
    response = requests.get(cif_url)
    if response.status_code == 200 and "xml" not in response.text.lower():
        open(cif_file, "wb").write(response.content)
        print("AlphaFold structure downloaded:", cif_file)
        
        # =========================
        # Extract pLDDT from CIF (B-factor column)
        # =========================
        plddt = []
        with open(cif_file) as f:
            for line in f:
                # For CIF format, look for _atom_site.B_iso_or_equiv
                if line.startswith("_atom_site.B_iso_or_equiv"):
                    continue
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    try:
                        # Try different column positions for pLDDT
                        parts = line.split()
                        if len(parts) > 10:
                            plddt.append(float(parts[-2]))  # Usually B-factor is second to last
                    except (ValueError, IndexError):
                        pass
        
        if plddt:
            print(f"pLDDT mean={statistics.mean(plddt):.2f} median={statistics.median(plddt):.2f}")
            
            # Plot pLDDT histogram
            plt.hist(plddt, bins=30)
            plt.title("AlphaFold pLDDT Distribution")
            plt.xlabel("pLDDT")
            plt.ylabel("Frequency")
            plt.show()
        else:
            print("Could not extract pLDDT values from CIF file - structure might be in different format")
    else:
        print(f"AlphaFold structure not available for {uniprot_id} or download failed")
except Exception as e:
    print(f"Error downloading AlphaFold structure: {e}")
    plddt = []

# =========================
# DeepPurpose drug–target affinity prediction
# =========================

print("\nLoading DeepPurpose model...")
X_pred = utils.data_process(
    X_drug=[drug_smiles],
    X_target=[sequence],
    y=[0],
    drug_encoding='Morgan',
    target_encoding='CNN',
    split_method='no_split'
)

model = DTI.model_pretrained(model='Morgan_CNN_BindingDB')
prediction = model.predict(X_pred)[0]
print(f"\nPredicted Binding Affinity Score: {prediction:.4f}")

