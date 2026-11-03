#!/usr/bin/env bash
# =========================
# Step 3: Energy minimization, NVT (100 ps), NPT (1 ns)
# =========================
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p run && cd run
GPU="-nb gpu -pme gpu -update gpu"

gmx grompp -f ../mdp/em.mdp -c ../build/solv_ions.gro -p ../build/topol.top -o em.tpr
gmx mdrun -deffnm em -nb gpu

gmx grompp -f ../mdp/nvt.mdp -c em.gro -r em.gro -p ../build/topol.top -o nvt.tpr
gmx mdrun -deffnm nvt $GPU

gmx grompp -f ../mdp/npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p ../build/topol.top -o npt.tpr
gmx mdrun -deffnm npt $GPU

echo "Equilibration done: run/npt.gro, run/npt.cpt"
