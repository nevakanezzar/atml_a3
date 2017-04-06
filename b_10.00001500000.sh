#$ -l tmem=4G
#$ -l h_vmem=4G
#$ -l h_rt=36:00:00
#$ -S /bin/bash
#$ -wd /home/skasewa/git/atml_a3/
hostname
date
python3 ~/git/atml_a3/b34_2.py 1 0.00001 500000
