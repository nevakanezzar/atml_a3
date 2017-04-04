#$ -l tmem=1G
#$ -l h_vmem=1G
#$ -l h_rt=12:00:00
#$ -S /bin/bash
#$ -wd /home/skasewa/git/atml_a3/
hostname
date
python3 ~/git/atml_a3/b2.py 1 0.01 500000
