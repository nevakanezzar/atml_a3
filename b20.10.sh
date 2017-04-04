#$ -l tmem=2G
#$ -l h_vmem=2G
#$ -l h_rt=12:00:00
#$ -S /bin/bash
#$ -wd /home/skasewa/git/atml_a3/
hostname
date
python3 ~/git/atml_a3/b2.py 2 0.1 0
