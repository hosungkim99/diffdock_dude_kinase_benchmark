
reinitialize
set ray_opaque_background, off
bg_color white

load /home/deepfold/users/hosung/dataset/DUD-E/dude_raw/wee1/receptor.pdb, prot
load /home/deepfold/users/hosung/dataset/DUD-E/dude_raw/wee1/results/actives/wee1_active_000083/rank1_confidence0.70.sdf, lig

hide everything
show cartoon, prot
color gray70, prot

# ligand styling
show sticks, lig
util.cbag lig

# define pocket residues around ligand
select pocket, byres (prot within 6.0 of lig)
show sticks, pocket
color tv_blue, pocket and elem C

# make view centered on ligand
zoom lig, 12
set cartoon_transparency, 0.2

# optional: add surface for pocket (comment/uncomment)
# show surface, pocket
# set transparency, 0.4, pocket

# rendering
set antialias, 2
set ray_trace_mode, 1
ray 2000, 1500
png /home/deepfold/users/hosung/dataset/DUD-E/dude_raw/wee1/eval/diffdock/figs_high_conf_actives/active_wee1_active_000083_conf0.70.png, dpi=300
quit
