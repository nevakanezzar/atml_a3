#Problem A, task 5
import sys
import numpy as np
import gym
import tensorflow as tf
import random
import time

#constants
EPSILON = 0.05
DISCOUNT = 0.99
NUM_EPISODES = 2000
MAX_EPISODE_LEN = 300
NUM_EPISODES_EVAL = 20
NUM_TRIALS = 1

#filenames
SAVE_FOLDER = './save/'
MODEL_FOLDER = '.'
TENSORBOARD_FOLDER = '.'


#initialise environment
gym.envs.register(
    id = 'CartPoleModified-v0',
    entry_point = 'gym.envs.classic_control:CartPoleEnv',
    max_episode_steps = MAX_EPISODE_LEN,
)
env = gym.make('CartPoleModified-v0')


#reproducibility initializations
SEED = 40
# tf.reset_default_graph()
# tf.set_random_seed(SEED)
# np.random.seed(SEED) 
# random.seed(SEED)
# env.seed(SEED) 


#function that modifies the output (usually reward) as per directions
def modify_outputs(obs, rew, ter, inf): 
	if ter == True:
		rew = -1
	else:
		rew = 0	
	return obs, rew, ter, inf


STATE_DIM = 4
ACTION_DIM = env.action_space.n  # number of possible actions
SCALAR_DIM = 1


#hyperparameters
LEARNING_RATE = 0.00001
LAMBDA = 0.0
HIDDEN_DIM = 30
STD = 0.01


#create q learning graph

#tf functions
#model 2: neural net with HIDDEN_DIM-unit hidden layer
def q_hat(state):
	w1 = tf.get_variable("weight1", shape=[STATE_DIM, HIDDEN_DIM], initializer=tf.truncated_normal_initializer(0.0,STD))
	b1 = tf.get_variable("bias1", shape=[HIDDEN_DIM], initializer=tf.truncated_normal_initializer(0.0,STD))
	w2 = tf.get_variable("weight2", shape=[HIDDEN_DIM, ACTION_DIM], initializer=tf.truncated_normal_initializer(0.0,STD))
	b2 = tf.get_variable("bias2", shape=[ACTION_DIM], initializer=tf.truncated_normal_initializer(0.0,STD))
	q = tf.matmul(tf.nn.relu(tf.matmul(state,w1) + b1), w2) + b2
	return q


#prediction inputs
s_in = tf.placeholder(tf.float32, [None,STATE_DIM])
a_in = tf.placeholder(tf.int32, [None])
r_in = tf.placeholder(tf.float32, [None])
s1_in = tf.placeholder(tf.float32, [None,STATE_DIM])
discount_in = tf.placeholder(tf.float32)
row_indices_in = tf.placeholder(tf.int32, [None])
not_done_in = tf.placeholder(tf.float32, [None])

#table of action indices
actions_indices = tf.stack([row_indices_in, a_in],axis=1)

#constants (inferred or otherwise)
batch_size = tf.shape(s_in)[0]

with tf.variable_scope("QFA") as scope:
	q_out = q_hat(s_in)
	scope.reuse_variables()
	q1_out = q_hat(s1_in)

# target = tf.cond(tf.equal(tf.reduce_mean(r_in), tf.constant(-1.0)),lambda:r_in, lambda: discount_in * tf.stop_gradient(tf.reduce_max(q1_out,axis=1)))

target = r_in + discount_in * not_done_in * tf.stop_gradient(tf.reduce_max(q1_out,axis=1))

bellman_residual = target - tf.gather_nd(q_out,actions_indices) #q_out[0][a_in[0]] #

thetas = [item for item in tf.trainable_variables()]
reg_losses = [LAMBDA * tf.nn.l2_loss(item) for item in tf.trainable_variables() if 'weight' in item.name]

loss = tf.reduce_mean(0.5*tf.square(bellman_residual)) + tf.reduce_sum(reg_losses)
train_op = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss)


def run():
	losses = np.zeros([NUM_TRIALS,NUM_EPISODES])
	bellman_losses = np.zeros([NUM_TRIALS,NUM_EPISODES])
	disc_rewards = np.zeros([NUM_TRIALS,NUM_EPISODES])
	aver_moves = np.zeros([NUM_TRIALS,NUM_EPISODES])

	for trial in range(NUM_TRIALS):
		init_op = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init_op)	# init vars
			for episode in range(NUM_EPISODES):
				s_t = env.reset()
				l1 = 0.0
				bellman_l1 = 0.0
				steps = 0
				#first we train:
				while 1:
					#select action
					if np.random.random()>EPSILON:
						qs = sess.run(q_out,feed_dict={s_in:np.expand_dims(s_t,axis=0)})
						a_t = np.argmax(qs)				
					else:
						a_t = np.random.choice(ACTION_DIM)
					steps+=1
					
					#take a step in the env 
					s_t1, r_t1, done, info = env.step(a_t)
					s_t1, r_t1, done, info = modify_outputs(s_t1, r_t1, done, info)
					not_done = not done

					#learn
					[l2,bellman_l2,_] = sess.run([loss,bellman_residual,train_op], feed_dict={
									s_in:np.array([s_t]),
									a_in:np.array([a_t]),
									r_in:np.array([r_t1]),
									s1_in:np.array([s_t1]),
									discount_in:DISCOUNT,
									row_indices_in:np.arange(1),
									not_done_in:np.array([not_done*1])
									})
					l1 += l2
					bellman_l1 += bellman_l2
					if done == True:
						break
					s_t = s_t1


				#then we evaluate:
				ave = 0
				time_per_episode = np.zeros(NUM_EPISODES_EVAL)
				reward_per_episode = np.zeros(NUM_EPISODES_EVAL)
				for episode_eval in range(NUM_EPISODES_EVAL):
					s_t = env.reset()
					episode_reward = 0.0
					cumulative_discount = 1.0
					for t in range(MAX_EPISODE_LEN):
						qs = sess.run(q_out,feed_dict={s_in:np.expand_dims(s_t,axis=0)})
						a_t = np.argmax(qs)
						# print(qs,a_t)
						s_t, r_t1, done, info = env.step(a_t)
						s_t, r_t1, done, info = modify_outputs(s_t, r_t1, done, info)
						episode_reward += cumulative_discoundt*r_t1
						cumulative_discount = cumulative_discount * DISCOUNT
						if info != {}: print(info)
						if done:
							break
					# print(t)
					time_per_episode[episode_eval] = t+1
					reward_per_episode[episode_eval] = episode_reward
				ave = np.mean(time_per_episode)
				rew = np.mean(reward_per_episode)
				if episode%20==0:
					print("Trial:",trial,"Episode", episode,"loss:",'{0:.4f}'.format(l1),"\tAverage performance over",NUM_EPISODES_EVAL,"evaluation trials: Moves",'{0:.3f}'.format(ave),"| Reward",'{0:.4f}'.format(rew))
				losses[trial,episode] = l1
				bellman_losses[trial,episode] = bellman_l1
				disc_rewards[trial,episode] = rew
				aver_moves[trial,episode] = ave
	

	# print("losses",losses)
	# print("bellman", bellman_losses)
	# print("disc_rew", disc_rewards)
	# print("aver_moves",aver_moves)

	concat = np.concatenate([losses,bellman_losses,disc_rewards,aver_moves])
	# print(concat)

	year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
	save_filename = SAVE_FOLDER+'a5plots_'+HIDDEN_DIM+"-"+hour+'_'+minute+'.csv'
	print(save_filename)
	with open(save_filename,'wb') as f:
		np.savetxt(save_filename, concat, fmt='%.5f',delimiter=",")

if __name__ == '__main__':
	run()






