#!/usr/bin/env bash
# =========================
# Step 4: Production MD (default 100 ns, set in mdp/md.mdp)
# =========================
# Usage: ./04_production.sh [name] [nsteps]
#   ./04_production.sh md_rep1            -> full 100 ns
#   ./04_production.sh md_test 1000000    -> 2 ns test run
set -euo pipefail
cd "$(dirname "$0")/run"
NAME=${1:-md}
NSTEPS=${2:--2}   # -2 = use nsteps from mdp

gmx grompp -f ../mdp/md.mdp -c npt.gro -t npt.cpt -p ../build/topol.top -o "$NAME.tpr"
gmx mdrun -deffnm "$NAME" -nsteps "$NSTEPS" -nb gpu -pme gpu -update gpu

echo "Done: run/$NAME.xtc"
