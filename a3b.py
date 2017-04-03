#Problem A, task 3b
import sys
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
SAVE_FOLDER = './save/'
MODEL_FOLDER = './model/'
TENSORBOARD_FOLDER = '.'

#initialise environment
gym.envs.register(
    id = 'CartPoleModified-v0',
    entry_point = 'gym.envs.classic_control:CartPoleEnv',
    max_episode_steps = MAX_EPISODE_LEN,
)
env = gym.make('CartPoleModified-v0')


#reproducibility initializations
SEED = 42
tf.reset_default_graph()
tf.set_random_seed(SEED)
np.random.seed(SEED) 
random.seed(SEED)
env.seed(SEED) 


#function that modifies the output (usually reward) as per directions
def modify_outputs(obs, rew, ter, inf): 
	if ter == True:
		rew = -1
	else:
		rew = 0	
	return obs, rew, ter, inf


# collect 2000 episodes under a uniform random policy
print("\ncollecting 2000 episodes under a uniform random policy...")
time_per_episode = np.zeros(NUM_EPISODES)
reward_per_episode = np.zeros(NUM_EPISODES)
experience = []
for episode in range(NUM_EPISODES):
	s_t = env.reset()
	episode_reward = 0.0
	cumulative_discount = DISCOUNT
	for t in range(MAX_EPISODE_LEN):
		a_t = env.action_space.sample()
		s_t1, r_t1, done, info = env.step(a_t)
		s_t1, r_t1, done, info = modify_outputs(s_t1, r_t1, done, info)
		experience.append([s_t,a_t,r_t1,s_t1])
		if done:
			break
		s_t = s_t1
print("collected",len(experience),"transitions...")


STATE_DIM = len(experience[0][0])
ACTION_DIM = env.action_space.n  # number of possible actions
SCALAR_DIM = 1

#preprocess experience into S, A, R, S1 matrices
T = len(experience)
S_orig = np.zeros([T,STATE_DIM],np.float)
A_orig = np.zeros([T,SCALAR_DIM],np.float)
R_orig = np.zeros([T,SCALAR_DIM],np.float)
S1_orig = np.zeros([T,STATE_DIM],np.float)
for i,_ in enumerate(experience):
	S_orig[i], A_orig[i], R_orig[i], S1_orig[i] = experience[i]
	# print(S[i], A[i], R[i], S1[i])
	A_orig = np.squeeze(A_orig)
	R_orig = np.squeeze(R_orig)

#hyperparameters
LEARNING_RATE = 0.5
LAMBDA = 0.0
HIDDEN_DIM = 100
NUM_EPOCHS = 500
MINI_BATCH_SIZE = 5000
MINI_BATCHES = T//MINI_BATCH_SIZE
STD = 0.00000000000000001
MODEL_FILENAME = MODEL_FOLDER+"a3b_"+str(LEARNING_RATE)+".model"

#create q learning graph


#tf functions

#model 1: linear transform
def q_hat(state):
	w1 = tf.get_variable("weight1", shape=[STATE_DIM, HIDDEN_DIM], initializer=tf.random_normal_initializer(0.0,STD))
	b1 = tf.get_variable("bias1", shape=[HIDDEN_DIM], initializer=tf.random_normal_initializer(0.0,STD))
	w2 = tf.get_variable("weight2", shape=[HIDDEN_DIM, ACTION_DIM], initializer=tf.random_normal_initializer(0.0,STD))
	b2 = tf.get_variable("bias2", shape=[ACTION_DIM], initializer=tf.random_normal_initializer(0.0,STD))
	q = tf.matmul(tf.nn.relu(tf.matmul(state,w1) + b1),w2) + b2
	# q = tf.matmul(tf.nn.relu(tf.matmul(state,w1)),w2)
	return q


#prediction inputs
s_in = tf.placeholder(tf.float32, [None,STATE_DIM])
a_in = tf.placeholder(tf.int32, [None])
r_in = tf.placeholder(tf.float32, [None])
s1_in = tf.placeholder(tf.float32, [None,STATE_DIM])
discount_in = tf.placeholder(tf.float32)
row_indices_in = tf.placeholder(tf.int32, [None])

#table of action indices
actions_indices = tf.stack([row_indices_in, a_in],axis=1)

#constants (inferred or otherwise)
batch_size = tf.shape(s_in)[0]

with tf.variable_scope("QFA") as scope:
	q_out = q_hat(s_in)
	scope.reuse_variables()
	q1_out = q_hat(s1_in)

A = (tf.add(r_in,tf.constant(1.0)))
bellman_residual = r_in + A * discount_in * tf.stop_gradient(tf.reduce_max(q1_out,axis=1)) - tf.gather_nd(q_out,actions_indices)

thetas = [item for item in tf.trainable_variables()]
reg_losses = [LAMBDA * tf.nn.l2_loss(item) for item in tf.trainable_variables() if 'weight' in item.name]

loss = 0.5*tf.reduce_mean(tf.square(bellman_residual)) + tf.reduce_sum(reg_losses)
train_op = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss)


def run():
	losses = np.zeros([NUM_TRIALS,NUM_EPOCHS])
	bellman_losses = np.zeros([NUM_TRIALS,NUM_EPOCHS])
	disc_rewards = np.zeros([NUM_TRIALS,NUM_EPOCHS])
	aver_moves = np.zeros([NUM_TRIALS,NUM_EPOCHS])
	last_t = None

	for trial in range(NUM_TRIALS):
		saver = tf.train.Saver(write_version = tf.train.SaverDef.V1)
		init_op = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init_op)  			# init vars
			for epoch in range(NUM_EPOCHS):
				#first we train:
				inds = np.random.choice(T,T,replace=False)
				# inds = np.arange(T)
				l1 = 0.0
				bellman_l1 = 0.0
				for i in range(MINI_BATCHES):
					S = S_orig[inds[i*MINI_BATCH_SIZE : (i+1)*MINI_BATCH_SIZE]]
					A = A_orig[inds[i*MINI_BATCH_SIZE : (i+1)*MINI_BATCH_SIZE]]
					R = R_orig[inds[i*MINI_BATCH_SIZE : (i+1)*MINI_BATCH_SIZE]]
					S1 = S1_orig[inds[i*MINI_BATCH_SIZE : (i+1)*MINI_BATCH_SIZE]]
				
					[_] = sess.run([train_op], feed_dict={s_in:S,a_in:A,r_in:R,s1_in:S1,discount_in:DISCOUNT,row_indices_in:np.arange(len(S))})
					# [l2,bellman_l2] = sess.run([loss,bellman_residual], feed_dict={s_in:S,a_in:A,r_in:R,s1_in:S1,discount_in:DISCOUNT,row_indices_in:np.arange(len(S))})
					# l1+=l2
					# bellman_l1 += np.mean(bellman_l2)

				[l1,bellman_l1] = sess.run([loss,bellman_residual], feed_dict={s_in:S_orig,a_in:A_orig,r_in:R_orig,s1_in:S1_orig,discount_in:DISCOUNT,row_indices_in:np.arange(len(S_orig))})
				bellman_l1 = np.mean(bellman_l1)
				# l1 = l1/MINI_BATCHES
				# bellman_l1 = bellman_l1/MINI_BATCHES

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
						episode_reward += cumulative_discount*r_t1
						cumulative_discount = cumulative_discount * DISCOUNT
						if info != {}: print(info)
						if done:
							break
					# print(t)
					time_per_episode[episode_eval] = t+1
					reward_per_episode[episode_eval] = episode_reward
				ave = np.mean(time_per_episode)
				rew = np.mean(reward_per_episode)
				if epoch%1==0:
					print("Trial:",trial,"Epoch", epoch,"loss:",'{0:.6f}'.format(l1),"\tAverage performance over",NUM_EPISODES_EVAL,"evaluation trials: Moves",'{0:.3f}'.format(ave),"| Reward",'{0:.4f}'.format(rew))

				losses[trial,epoch] = l1
				bellman_losses[trial,epoch] = bellman_l1
				disc_rewards[trial,epoch] = rew
				aver_moves[trial,epoch] = ave
			saver.save(sess,MODEL_FILENAME)
			print("Saved model at",MODEL_FILENAME)
		

	# print("losses",losses)
	# print("bellman", bellman_losses)
	# print("disc_rew", disc_rewards)
	# print("aver_moves",aver_moves)

	concat = np.concatenate([losses,bellman_losses,disc_rewards,aver_moves])
	# print(concat)

	year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
	save_filename = SAVE_FOLDER+'a3bplots_'+str(LEARNING_RATE)+'_'+hour+'_'+minute+'.csv'
	print(save_filename)
	with open(save_filename,'wb') as f:
		np.savetxt(save_filename, concat, fmt='%.5f',delimiter=",")



if __name__ == '__main__':
	run()

