#$ -l tmem=2G
#$ -l h_vmem=2G
#$ -l h_rt=36:00:00
#$ -S /bin/bash
#$ -wd /home/skasewa/git/atml_a3/
hostname
date
python3 ~/git/atml_a3/b34.py 1 0.01 500000
