#Problem B, task 1
import sys
import numpy as np
import gym
import random
import time
from PIL import Image
import select


try:
	GAME_INDEX = int(sys.argv[1])
	NUM_EPISODES = int(sys.argv[2])
except:
	print("Something went wrong. Signature is game_index num_episodes [seed]")
	sys.exit()

#constants
GAMES = ['Pong-v3','MsPacman-v3','Boxing-v3']
DISCOUNT = 0.99
FRAMES = 4

#initialise environment
game = GAMES[GAME_INDEX]
env = gym.make(game)
ACTION_DIM = env.action_space.n
print('Game : '+game)
print('Possible actions : {}'.format(ACTION_DIM))

#reproducibility initializations
if len(sys.argv) == 4:
	SEED = min(int(sys.argv[3]),0)
	env.seed(SEED) 
	np.random.seed(SEED) 
	random.seed(SEED)


#hyperparameters
#NONE

#file names
#NONE


def run():
	inp2 = ''
	test_steps = np.zeros(NUM_EPISODES)
	test_d_rew = np.zeros(NUM_EPISODES)
	test_t_rew = np.zeros(NUM_EPISODES)
	for j in range(NUM_EPISODES):
		e_ep_t_rew = 0
		e_ep_d_rew = 0
		e_ep_steps = 1
		e_s_t = env.reset()	
		while 1:
			
			#render environment if 'r' key entered
			inp,_,_ = select.select([sys.stdin],[],[],0)
			for s in inp:
				if s == sys.stdin:
					inp2 = sys.stdin.readline()
					inp2 = inp2[0].lower()
			if inp2=='r': 
				env.render()
			else:
				env.render(close=True)

			e_a_t = np.random.choice(ACTION_DIM)	
			e_s1_t, e_r_t, e_done, e_info = env.step(e_a_t)
			
			e_ep_t_rew += e_r_t
			e_ep_d_rew += e_r_t * DISCOUNT**e_ep_steps
			e_ep_steps += 1
			if e_done:
				break

		test_steps[j] = e_ep_steps
		test_d_rew[j] = e_ep_d_rew
		test_t_rew[j] = e_ep_t_rew

		e_avg_steps = np.mean(test_steps)
		e_avg_d_rew = np.mean(test_d_rew)
		e_avg_t_rew = np.mean(test_t_rew)

		e_std_steps = np.std(test_steps)
		e_std_d_rew = np.std(test_d_rew)
		e_std_t_rew = np.std(test_t_rew)

	print("Evaluation over",NUM_EPISODES,"episodes")
	print("Avg steps:",e_avg_steps,"Standard deviation:",e_std_steps)
	print("Avg tot rew:",e_avg_t_rew,"Standard deviation:",e_std_t_rew)
	print("Avg disc rew:",e_avg_d_rew,"Standard deviation:",e_std_d_rew)
	
	

run()

