# =========================
# PyMOL: ABL1 kinase + Dasatinib MD trajectory
# =========================
# Run from week6/md:  pymol view.pml
# Input from ./05_pymol_prep.sh md_test

load run/md_test_view.pdb, complex
load_traj run/md_test_view.xtc, complex, state=1
smooth complex, 30, 3

hide everything
bg_color white
set cartoon_transparency, 0.15
set stick_radius, 0.18

# Protein
show cartoon, polymer
color grey80, polymer
color lightblue, polymer and resi 248-256           # P-loop
color palecyan, polymer and resi 316-321           # hinge
color lightpink, polymer and resi 381-402          # activation loop (DFG 381-383)

# Dasatinib
show sticks, resn LIG and not (elem H and neighbor elem C)
util.cbag resn LIG

# Pocket residues within 4 A of the drug (first frame)
select pocket, byres (polymer within 4 of resn LIG) and not name N+C+O
show sticks, pocket and not elem H
util.cbaw pocket
util.cbay polymer and resi 315+318 and not name N+C+O
show sticks, polymer and resi 315+318 and not name N+C+O and not elem H
label polymer and resi 315 and name CA, "Thr315 (gatekeeper)"
label polymer and resi 318 and name CA, "Met318 (hinge)"
set label_size, 18
set label_color, black

# Hydrogen bonds drug <-> protein, recomputed in every frame
set dynamic_measures, 1
distance hbonds, resn LIG, polymer, mode=2, state=0
color orange, hbonds
hide labels, hbonds

deselect
orient resn LIG
zoom resn LIG, 9
mplay
