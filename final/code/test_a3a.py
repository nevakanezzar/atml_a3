#Problem A, task 3a -- test
from loadmodels import LoadModels
import sys
import select
import numpy as np
import gym
import tensorflow as tf
import random
import time

#constants
DISCOUNT = 0.99
NUM_EPISODES = 2000
MAX_EPISODE_LEN = 300
NUM_EPISODES_EVAL = 10
NUM_TRIALS = 1

#filenames
MODEL_FOLDER = '../model/'
SAVE_FOLDER = MODEL_FOLDER
TENSORBOARD_FOLDER = '.'

#initialise environment
gym.envs.register(
    id = 'CartPoleModified-v0',
    entry_point = 'gym.envs.classic_control:CartPoleEnv',
    max_episode_steps = MAX_EPISODE_LEN,
)
env = gym.make('CartPoleModified-v0')


#reproducibility initializations
SEED = 189
# tf.reset_default_graph()
# tf.set_random_seed(SEED)
# np.random.seed(SEED) 
# random.seed(SEED)
# env.seed(SEED) 


#function that modifies the output (usually reward) as per directions
def modify_outputs(obs, rew, ter, inf, steps): 
	if steps == MAX_EPISODE_LEN:
		rew = 0
	elif ter == True:
		rew = -1
	else:
		rew = 0	
	return obs, rew, ter, inf



STATE_DIM = 4
ACTION_DIM = env.action_space.n  # number of possible actions
SCALAR_DIM = 1


#hyperparameters
try:
	LEARNING_RATE = 0.0 # float(sys.argv[1])
except:
	print("Something went wrong! Try providing a learning rate.")
	sys.exit(0)
LAMBDA = 0.0
HIDDEN_DIM = 100
NUM_EPOCHS = 500
MINI_BATCH_SIZE = 5000
MINI_BATCHES = MINI_BATCH_SIZE
STD = 0.00000000000000001

#file names
# year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
# MODEL = "a3a_"+sys.argv[1]+"_"+hour+'_'+minute
# MODEL_FILENAME = MODEL_FOLDER+MODEL+".model"
# SAVE_FILENAME = SAVE_FOLDER+MODEL+".csv"



#create q learning graph

#tf functions

#model 1: linear transform	
def q_hat(state):
	w1 = tf.get_variable("weight1", shape=[STATE_DIM, ACTION_DIM], dtype=tf.float64, initializer=tf.random_normal_initializer(0.0,STD))
	q = tf.matmul(state,w1)
	return q


#prediction inputs
s_in = tf.placeholder(tf.float64, [None,STATE_DIM])
a_in = tf.placeholder(tf.int32, [None])
r_in = tf.placeholder(tf.float64, [None])
s1_in = tf.placeholder(tf.float64, [None,STATE_DIM])
discount_in = tf.placeholder(tf.float64)
row_indices_in = tf.placeholder(tf.int32, [None])

#table of action indices
actions_indices = tf.stack([row_indices_in, a_in],axis=1)

#constants (inferred or otherwise)
batch_size = tf.shape(s_in)[0]

with tf.variable_scope("QFA") as scope:
	q_out = q_hat(s_in)
	scope.reuse_variables()
	q1_out = q_hat(s1_in)

A = (tf.add(r_in,tf.constant(1.0,dtype=tf.float64)))
bellman_residual = r_in + A * discount_in * tf.stop_gradient(tf.reduce_max(q1_out,axis=1)) - tf.gather_nd(q_out,actions_indices)

thetas = [item for item in tf.trainable_variables()]
reg_losses = [LAMBDA * tf.nn.l2_loss(item) for item in tf.trainable_variables() if 'weight' in item.name]

loss = 0.5*tf.reduce_mean(tf.square(bellman_residual)) + tf.reduce_sum(reg_losses)
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)



def load(LOAD_FILENAME):
	print("\n\n\n\nEvaluating model:",LOAD_FILENAME)
	inp2 = ''
	EVAL_TRIALS = 100
	with tf.Session() as sess:
		saver = tf.train.Saver(write_version = tf.train.SaverDef.V1)  
		saver.restore(sess,LOAD_FILENAME)
		ave = 0
		time_per_episode = np.zeros(EVAL_TRIALS)
		reward_per_episode = np.zeros(EVAL_TRIALS)
		for episode_eval in range(EVAL_TRIALS):
			s_t = env.reset()
			episode_reward = 0.0
			cumulative_discount = 1.0
			for t in range(MAX_EPISODE_LEN):

				inp,_,_ = select.select([sys.stdin],[],[],0)
				for s in inp:
					if s == sys.stdin:
						inp2 = sys.stdin.readline()
						inp2 = inp2[:-1].lower()

				qs = sess.run(q_out,feed_dict={s_in:np.expand_dims(s_t,axis=0)})
				a_t = np.argmax(qs)
				# print(qs,a_t)
				s_t, r_t1, done, info = env.step(a_t)
				s_t, r_t1, done, info = modify_outputs(s_t, r_t1, done, info, t+1)
				if inp2 == 'r':
					env.render()
				else:
					env.render(close=True)
				episode_reward += cumulative_discount*r_t1
				cumulative_discount = cumulative_discount * DISCOUNT
				if info != {}: print(info)
				if done:
					break
			# print(t)
			time_per_episode[episode_eval] = t+1
			reward_per_episode[episode_eval] = episode_reward
			print("Trial:",episode_eval,"\tMoves",'{0:.3f}'.format(t+1),"| Reward",'{0:.4f}'.format(episode_reward))
		ave = np.mean(time_per_episode)
		rew = np.mean(reward_per_episode)
		print("Average performance over",EVAL_TRIALS,"evaluation trials: Moves",'{0:.3f}'.format(ave),"| Reward",'{0:.4f}'.format(rew))



if __name__ == '__main__':
	stub = 'a3a'
	models = LoadModels(stub)	
	print("\nYou are about to run 100 evaluations of these trained models on CartPole-v0.")
	print("At any time during the evaluation, you may type 'r' and hit return to see the environment being rendered.")
	print("Enter any other key besides 'r' to stop the rendering.")
	x = input("Press Enter to continue...")
	for model in models:
		load(MODEL_FOLDER+model)

