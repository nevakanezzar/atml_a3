#Problem B, task 1
import sys
import numpy as np
import gym
import tensorflow as tf
import random
import time


#arguments
GAME_INDEX = 0

#constants
GAMES = ['Pong-v3','MsPacman-v3','Boxing-v3']
DISCOUNT = 0.99
EPSILON = 0.05
NUM_EPISODES = 1


#initialise environment
game = GAMES[GAME_INDEX]
env = gym.make(game)
STATE_DIM = None
SCALAR_DIM = 1
ACTION_DIM = env.action_space.n
print('Game : '+game)
print('Possible actions : {}'.format(ACTION_DIM))

#filenames
SAVE_FOLDER = './save/'
MODEL_FOLDER = './model/'
TENSORBOARD_FOLDER = '.'


#reproducibility initializations
SEED = 40
# tf.reset_default_graph()
# tf.set_random_seed(SEED)
# np.random.seed(SEED) 
# random.seed(SEED)
# env.seed(SEED) 


#function that modifies the output (usually reward) as per directions
def modify_outputs(obs, rew, ter, inf): 
	if rew <= -1:
		rew = -1
	elif rew >= 1:
		rew = 1
	else:
		rew = 0
	return obs, rew, ter, inf



#hyperparameters


#file names
MODEL = "b1"
MODEL_FILENAME = MODEL_FOLDER+MODEL+".model"
year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
SAVE_FILENAME = SAVE_FOLDER+MODEL+"_"+hour+'_'+minute+'.csv'



def runGame(numberOfEpisodes,DISCOUNT):
	numberOfFrames =cumulativeScores = []
	for episode in range(numberOfEpisodes):
		if episode == numberOfEpisodes - 1 :
			print('episode {}'.format(episode))
		else:
			print('episode {}'.format(episode), end = '\r')

		s_t = env.reset()	
		index = 0
		tempScore = 0
		while 1:
			a_t = np.random.choice(ACTION_DIM)
			s_t, r_t, done, info = env.step(a_t)
			s_t, r_t, done, info = modify_outputs(s_t, r_t, done, info)
			tempScore += r_t * DISCOUNT**index
			index += 1
			if done:
				break

		cumulativeScores.append(tempScore)
		numberOfFrames.append(index)

	frameMean = np.mean(numberOfFrames)
	frameStd = np.std(numberOfFrames)
	scoreMean = np.mean(cumulativeScores)
	scoreStd = np.std(cumulativeScores)
	print(s_t, s_t.shape)
	print('Mean Frame Count : {} -  Frame Count STD {}'.format(frameMean,frameStd) )
	print('Mean Score : {} -  Score STD {}'.format(scoreMean,scoreStd) )

	return frameMean, frameMean, scoreMean, scoreMean


runGame(NUM_EPISODES,DISCOUNT)