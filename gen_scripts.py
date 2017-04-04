
games = ['0','1','2']
learning_rates = ['0.1','0.01','0.001','0.0001','0.00001']
discount_factor = ['0','500000']

files = []

for g in games:
	for l in learning_rates:
		for d in discount_factor:
			filename = "b"+g+l+d+".sh"
			text = ["#$ -l tmem=1G\n",
					"#$ -l h_vmem=1G\n",
					"#$ -l h_rt=12:00:00\n",
					"#$ -S /bin/bash\n",
					"#$ -j y\n",
					"#$ -wd $HOME/git/atml_a3/\n"
					"hostname\n",
					"date\n",
					"python3 ~/git/atml_a3/b2.py "+g+" "+l+" "+d+"\n"]
			with open(filename,'w') as f:
				f.write("".join(text))	
			files.append(filename)

master = "masterscript.sh"

with open(master,'w') as f:
	for f1 in files:
		f.write("qsub "+f1+"\n")

