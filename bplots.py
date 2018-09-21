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

def LoadFiles(stub):
	dir_list = os.listdir(MODEL_FOLDER)
	model_list = []
	print("\nModels:")
	# print(dir_list)
	for f in dir_list:
		if stub in f:
			model_list.append(f)

	return model_list

#training plots
X_LABEL = 'Episodes'
labels = ['Bellman Loss','Average distance from target']
files = LoadFiles("train")
files = [f for f in files if '.csv' in f]
print(files)
for f1 in files:
	filename = './save/'+f1
	savename = './save/'+f1[:-4]+'.png'
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
			axs[j].plot(x,data[i*N_COLS + j],linewidth=0.5, color='#0000FF')
			axs[j].set_title(labels[i*N_COLS + j], fontsize=TITLE_FONT_SIZE)
			for tick in axs[j].yaxis.get_major_ticks():
				tick.label.set_fontsize(TICK_FONT_SIZE)
			for tick in axs[j].xaxis.get_major_ticks():
				tick.label.set_fontsize(TICK_FONT_SIZE)
			axs[j].set_xlabel(X_LABEL, fontsize=TICK_FONT_SIZE)

	fig.tight_layout() 
	# fig.show()
	fig.savefig(savename, dpi=300, bbox_inches='tight')
	# z = input()
	fig.close()



#test plots
X_LABEL = 'Every 50K steps'
labels = ['Average moves','Discounted Reward']
files = LoadFiles("test")
files = [f for f in files if '.csv' in f]
print(files)

for f1 in files:
	filename = './save/'+f1
	savename = './save/'+f1[:-4]+'.png'
	print(filename,savename)

	#read data
	data = np.loadtxt(filename,delimiter=',')

	episodesx2, trials = data.shape
	episodes = int(episodesx2 / 2)
	data2 = []
	data = data.reshape([2, episodes, trials])
	for i in range(episodes):
		if i != 0 and i != episodes-2:
			data2.append(data[:,i,:])

	data2 = np.array(data2)
	n,episodes,trials = data2.shape
	# print(n,episodes,trials)
	data2 = np.swapaxes(data2,1,0)
	n,episodes,trials = data2.shape
	data_means = np.mean(data2,axis=2)
	data_std = np.std(data2,axis=2)
	# print(data_means, data_std)

	x = np.arange(0,episodes,step,dtype=np.int32)
	fig, axs = plt.subplots(nrows=N_ROWS, ncols=N_COLS, sharex=True)

	for i in range(N_ROWS):
		for j in range(N_COLS):
			axs[j].plot(x,data_means[i*N_COLS + j,x],linewidth=0.5, color='#0000FF')
			# axs[j].fill_between(x,data_means[i*N_COLS + j,x] - data_std[i*N_COLS + j,x],data_means[i*N_COLS + j,x] + data_std[i*N_COLS + j,x], antialiased=True, facecolor='#A0BFFF')
			axs[j].set_title(labels[i*N_COLS + j], fontsize=TITLE_FONT_SIZE)
			for tick in axs[j].yaxis.get_major_ticks():
				tick.label.set_fontsize(TICK_FONT_SIZE)
			for tick in axs[j].xaxis.get_major_ticks():
				tick.label.set_fontsize(TICK_FONT_SIZE)
			axs[j].set_xlabel(X_LABEL, fontsize=TICK_FONT_SIZE)
	fig.tight_layout() 
	# fig.show()
	fig.savefig(savename, dpi=300, bbox_inches='tight')
	# z = input()
	fig.close()




