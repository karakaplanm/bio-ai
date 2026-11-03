#!/usr/bin/env bash
# =========================
# Step 2: Build solvated ABL1 + Dasatinib system for GROMACS
# =========================
# Requires: python3 01_prepare.py (build/protein.pdb, build/lig.mol2)
# Force fields: amber14sb (protein), GAFF2 + AM1-BCC via acpype (Dasatinib), TIP3P water
set -euo pipefail
cd "$(dirname "$0")/build"

# Ligand parameters (AM1-BCC charges take a few minutes)
[ -d LIG.acpype ] || acpype -i lig.mol2 -b LIG -c bcc -a gaff2 -n 0 -o gmx

# Protein topology
gmx pdb2gmx -f protein.pdb -o protein.gro -p topol.top -i posre_protein.itp \
    -ff amber14sb -water tip3p -ignh

# Ligand: atomtypes must come before any moleculetype, so split them out
python3 - <<'EOF'
itp = open("LIG.acpype/LIG_GMX.itp").read()
start = itp.index("[ atomtypes ]")
end = itp.index("[ moleculetype ]")
open("lig_atomtypes.itp", "w").write(itp[start:end])
open("lig.itp", "w").write(itp[:start] + itp[end:]
     + '\n#ifdef POSRES\n#include "LIG.acpype/posre_LIG.itp"\n#endif\n')

top = open("topol.top").read()
ff = '#include "amber14sb.ff/forcefield.itp"\n'
top = top.replace(ff, ff + '#include "lig_atomtypes.itp"\n')
water = "; Include water topology"
top = top.replace(water, '; Include ligand topology\n#include "lig.itp"\n\n' + water)
top = top.rstrip("\n") + "\nLIG                 1\n"
open("topol.top", "w").write(top)

# Complex coordinates: protein + ligand in one .gro
prot = open("protein.gro").read().splitlines()
lig = open("LIG.acpype/LIG_GMX.gro").read().splitlines()
atoms = prot[2:-1] + lig[2:-1]
open("complex.gro", "w").write("\n".join([prot[0], str(len(atoms))] + atoms + [prot[-1]]) + "\n")
EOF

# Box, water, ions (neutralize + 0.15 M NaCl)
gmx editconf -f complex.gro -o box.gro -c -d 1.0 -bt dodecahedron
gmx solvate -cp box.gro -cs spc216.gro -o solv.gro -p topol.top
# maxwarn 1: net charge warning, removed by genion below
gmx grompp -f ../mdp/em.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 1
echo SOL | gmx genion -s ions.tpr -o solv_ions.gro -p topol.top \
    -pname NA -nname CL -neutral -conc 0.15

echo "System ready: build/solv_ions.gro + build/topol.top"
