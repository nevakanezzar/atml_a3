import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

#constants
N_ROWS = 1
N_COLS = 2
TICK_FONT_SIZE = 8
TITLE_FONT_SIZE = 10
step = 1

MODEL_FOLDER = './save/'


#training plots
X_LABEL = 'Episodes'
labels = ['Bellman Loss','Average distance from target']
files = ['b34_0_0.00001_0_05_20train.csv']
# files = [f for f in files if '.csv' in f]
print(files)
for f1 in files:
	filename = './save/'+f1
	savename = './save/'+f1[:-4]+'-1.png'
	print(filename,savename)

	#read data
	data = np.loadtxt(filename,delimiter=',')

	trialsx2 = data.shape
	# print(trialsx2)
	episodes = int(trialsx2[0] / 2)

	# print(data)
	data = data.reshape([2, episodes])
	# print(data)
	# data is now in the form:
	# losses 
	# bellman_losses
	
	data_means = np.mean(data,axis=1)
	data_std = np.std(data,axis=1)
	# print(data_means, data_std)

	x = np.arange(0,episodes,step,dtype=np.int32)
	fig, axs = plt.subplots(nrows=N_ROWS, ncols=N_COLS, sharex=True)

	for i in range(N_ROWS):
		for j in range(N_COLS):
			axs[j].plot(x[1:],data[i*N_COLS + j][1:],linewidth=0.5, color='#0000FF')
			axs[j].set_title(labels[i*N_COLS + j], fontsize=TITLE_FONT_SIZE)
			for tick in axs[j].yaxis.get_major_ticks():
				tick.label.set_fontsize(TICK_FONT_SIZE)
			for tick in axs[j].xaxis.get_major_ticks():
				tick.label.set_fontsize(TICK_FONT_SIZE)
			axs[j].set_xlabel(X_LABEL, fontsize=TICK_FONT_SIZE)

	fig.tight_layout() 
	fig.show()
	fig.savefig(savename, dpi=300, bbox_inches='tight')
	z = input()
	fig.close()

