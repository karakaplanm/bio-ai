#!/usr/bin/env bash
# =========================
# Step 5: Make a whole, centered, fitted protein+ligand trajectory for PyMOL
# =========================
# Usage: ./05_pymol_prep.sh [name]   (default md_test)
set -euo pipefail
cd "$(dirname "$0")/run"
NAME=${1:-md_test}

# Index group "Protein_LIG"
printf '1 | 13\nq\n' | gmx make_ndx -f "$NAME.tpr" -o index.ndx
GRP=Protein_LIG

# Keep molecules whole and centered, then fit on protein backbone
printf 'Protein\n%s\n' "$GRP" | gmx trjconv -s "$NAME.tpr" -f "$NAME.xtc" -n index.ndx \
    -o tmp_center.xtc -pbc mol -center -ur compact
printf 'Backbone\n%s\n' "$GRP" | gmx trjconv -s "$NAME.tpr" -f tmp_center.xtc -n index.ndx \
    -o "${NAME}_view.xtc" -fit rot+trans
printf 'Protein\n%s\n' "$GRP" | gmx trjconv -s "$NAME.tpr" -f "$NAME.xtc" -n index.ndx \
    -o "${NAME}_view.pdb" -pbc mol -center -dump 0
rm -f tmp_center.xtc

echo "PyMOL input: run/${NAME}_view.pdb + run/${NAME}_view.xtc"
