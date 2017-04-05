#plotter-1
import sys
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

#constants
N_ROWS = 2
N_COLS = 2
TICK_FONT_SIZE = 8
TITLE_FONT_SIZE = 10

#set hyperparams
step = 1
f = 'a3aplots_1e-05_22_56'

if len(sys.argv) > 1:
	f = sys.argv[1]

if len(sys.argv) > 2:
	step = int(sys.argv[2])

filename = './save/'+f+'.csv'
savename = './save/'+f+'.png'


#read data
data = np.loadtxt(filename,delimiter=',')

trialsx4, episodes = data.shape
trials = int(trialsx4 / 4)

print(data)
data = data.reshape([4, trials, episodes])
print(data)
# data is now in the form:
# losses 
# bellman_losses
# disc_rewards 
# aver_moves 
labels = ['Bellman Loss','Average distance from target','Discounted rewards','Average moves']


data_means = np.mean(data,axis=1)
data_std = np.std(data,axis=1)
print(data_means, data_std)

x = np.arange(0,episodes,step,dtype=np.int32)
fig, axs = plt.subplots(nrows=N_ROWS, ncols=N_COLS, sharex=True)

for i in range(N_ROWS):
	for j in range(N_COLS):
		axs[i,j].plot(x,data_means[i*N_COLS + j,x],linewidth=0.5, color='#0000FF')
		axs[i,j].fill_between(x,data_means[i*N_COLS + j,x] - data_std[i*N_COLS + j,x],data_means[i*N_COLS + j,x] + data_std[i*N_COLS + j,x], antialiased=True, facecolor='#A0BFFF')
		axs[i,j].set_title(labels[i*N_COLS + j], fontsize=TITLE_FONT_SIZE)
		for tick in axs[i,j].yaxis.get_major_ticks():
			tick.label.set_fontsize(TICK_FONT_SIZE)
		for tick in axs[i,j].xaxis.get_major_ticks():
			tick.label.set_fontsize(TICK_FONT_SIZE)
		if i==1:
			axs[i,j].set_xlabel('Epochs', fontsize=TICK_FONT_SIZE)

fig.tight_layout() 
fig.show()
fig.savefig(savename, dpi=300, bbox_inches='tight')
z = input()