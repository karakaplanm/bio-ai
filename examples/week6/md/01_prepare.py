# =========================
# Step 1: Prepare ABL1 kinase + Dasatinib from PDB 2GQG
# =========================
# https://www.rcsb.org/structure/2GQG
# Output:
#   build/protein.pdb  -> chain A, heavy atoms only (hydrogens added later by gmx pdb2gmx)
#   build/lig.mol2     -> Dasatinib with hydrogens, ready for acpype (GAFF2)

import os
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from rdkit import Chem
from rdkit.Chem import AllChem

pdb_in = "input/2GQG.pdb"
os.makedirs("build", exist_ok=True)

# Neutral Dasatinib; piperazine pKa ~6.8, so mostly neutral at pH 7.4
dasatinib_smiles = "CC1=NC(=CC(=N1)NC2=NC=C(S2)C(=O)NC3=C(C=CC=C3Cl)C)N4CCN(CC4)CCO"

# =========================
# Protein: chain A only
# =========================
fixer = PDBFixer(filename=pdb_in)
chain_ids = [c.id for c in fixer.topology.chains()]
fixer.removeChains([i for i, cid in enumerate(chain_ids) if cid != "A"])

# PTR393 has no resolved phosphate in chain A -> treat as plain tyrosine
fixer.findNonstandardResidues()
print("Nonstandard residues:", [(r.name, r.id, rep) for r, rep in fixer.nonstandardResidues])
fixer.replaceNonstandardResidues()

fixer.removeHeterogens(keepWater=False)  # ligand is handled separately below
fixer.findMissingResidues()
fixer.missingResidues = {}  # do not model expression tag / termini
fixer.findMissingAtoms()
print("Residues with missing heavy atoms:", len(fixer.missingAtoms))
fixer.addMissingAtoms()

with open("build/protein.pdb", "w") as f:
    PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
print("Wrote build/protein.pdb")

# =========================
# Ligand: Dasatinib (1N1) from chain A
# =========================
lig_lines = [l for l in open(pdb_in)
             if l.startswith("HETATM") and l[17:20] == "1N1" and l[21] == "A"]
lig = Chem.MolFromPDBBlock("".join(lig_lines) + "END\n", removeHs=True)
template = Chem.MolFromSmiles(dasatinib_smiles)
lig = AllChem.AssignBondOrdersFromTemplate(template, lig)
lig = Chem.AddHs(lig, addCoords=True)
print("Ligand atoms (with H):", lig.GetNumAtoms(), "| formal charge:", Chem.GetFormalCharge(lig))

# Unique atom names + residue name LIG for acpype/GROMACS
counts = {}
for atom in lig.GetAtoms():
    el = atom.GetSymbol()
    counts[el] = counts.get(el, 0) + 1
    info = Chem.AtomPDBResidueInfo(f"{el}{counts[el]}".ljust(4), residueName="LIG", residueNumber=1)
    atom.SetMonomerInfo(info)

Chem.MolToMolFile(lig, "build/lig.mol")
os.system("obabel build/lig.mol -O build/lig.mol2 2>/dev/null")

# obabel keeps atom order but drops names -> write unique names and residue LIG
names = [a.GetMonomerInfo().GetName().strip() for a in lig.GetAtoms()]
out, section, i = [], None, 0
for line in open("build/lig.mol2"):
    if line.startswith("@<TRIPOS>"):
        section = line.strip()
    elif section == "@<TRIPOS>MOLECULE" and line.strip() == "*****":
        line = "LIG\n"
    elif section == "@<TRIPOS>ATOM" and line.strip():
        f = line.split()
        line = f"{f[0]:>7} {names[i]:<6} {f[2]:>10} {f[3]:>10} {f[4]:>10} {f[5]:<6} 1  LIG {f[8]:>10}\n"
        i += 1
    out.append(line)
open("build/lig.mol2", "w").writelines(out)
print("Wrote build/lig.mol2")
