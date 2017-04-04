#$ -l tmem=4G
#$ -l h_vmem=1G
#$ -l h_rt=12:00:00
#$ -S /bin/bash
#$ -wd /home/skasewa/git/atml_a3/
#$ -pe smp 4
hostname
date
python3 ~/git/atml_a3/b2.py 0 0.01 500000
