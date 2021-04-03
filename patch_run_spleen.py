# wrapper function for patch generation script
import subprocess
import os

# directorties for slides, masks and output patches should be written here

NEIDL = "/projectnb2/neidlpathology/Natural history folder/IRF Frederick/Spleen/"
MASKS = "/projectnb2/vkolagrp/ebola/data/masks/spleen_clean/"
PATCHES = "/projectnb2/vkolagrp/ebola/data/spleen_1000_lv2/

svs_files = []
for path, subdirs, files in os.walk(NEIDL):
    for name in files:
        svs_files.append(os.path.join(path, name))
        
mask_files = []
for path, subdirs, files in os.walk(MASKS):
    for name in files:
        mask_files.append(os.path.join(path, name))

# whole slide labels for each case

pre = ['1082', '1819', '1856', '0223', '0449', '0850']
mid = ['1834', '0717', '1849', '0452', '1325', '0522']
lte = ['1074', '1803', '1565', '0451', '0787', '1777', '0561', '0710', '0700', '1818', '1779', '1790', '0917', '1423', '1639' ]


dest_dirs = []
match_mask = []
match_svs = []
coverage = []

for file in svs_files:
   
    if any(p in file for p in pre) :
        for m in mask_files:
            if m.split('/')[-1].split('_')[0] in file:
                match_mask.append(m)
        match_svs.append(file)
        dest_dirs.append(PATCHES+'0_pre')
        coverage.append('.9')
        
    if any(m in file for m in mid) :
        for m in mask_files:
            if m.split('/')[-1].split('_')[0] in file:
                match_mask.append(m)
        match_svs.append(file)
        dest_dirs.append(PATCHES+'1_mid')
        coverage.append('.9')
        
    if any(l in file for l in lte) :
        for m in mask_files:
            if m.split('/')[-1].split('_')[0] in file:
                match_mask.append(m)
        match_svs.append(file)
        dest_dirs.append(PATCHES+'2_lte')
        coverage.append('.9')
        
print(*coverage, sep = "\n")
print(*match_mask, sep = "\n")
print(*match_svs, sep = "\n")
print(*dest_dirs, sep = "\n")

for i in range(len(match_svs)):
    print(i)
    subprocess.call([ 'python', 'patchgen_args_full_norm_spleen.py', '--s', match_svs[i], '--d', dest_dirs[i], '--o', '0', '--p', '1000', '--m', match_mask[i], '--c', coverage[i] ])