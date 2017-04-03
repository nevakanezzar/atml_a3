#Problem A, task 7
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
LEARNING_RATE = 0.0001
LAMBDA = 0.0
HIDDEN_DIM = 100
STD = 0.00001
BUFFER_SIZE = 2000
MINI_BATCH_SIZE = 512
MODULO = 5


#file names
MODEL = "a7_"+str(MODULO)
MODEL_FILENAME = MODEL_FOLDER+MODEL+".model"
year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
SAVE_FILENAME = SAVE_FOLDER+MODEL+"_"+hour+'_'+minute+'.csv'


#create q learning graph

#tf functions

#model 2: neural net with HIDDEN_DIM-unit hidden layer
w1 = tf.get_variable("weight1", shape=[STATE_DIM, HIDDEN_DIM], initializer=tf.truncated_normal_initializer(0.0,STD))
w2 = tf.get_variable("weight2", shape=[HIDDEN_DIM, ACTION_DIM], initializer=tf.truncated_normal_initializer(0.0,STD))

w3 = tf.get_variable("weight3", shape=[STATE_DIM, HIDDEN_DIM], initializer=tf.truncated_normal_initializer(0.0,STD))
w4 = tf.get_variable("weight4", shape=[HIDDEN_DIM, ACTION_DIM], initializer=tf.truncated_normal_initializer(0.0,STD))

#prediction inputs
s_in = tf.placeholder(tf.float32, [None,STATE_DIM])
a_in = tf.placeholder(tf.int32, [None])
r_in = tf.placeholder(tf.float32, [None])
s1_in = tf.placeholder(tf.float32, [None,STATE_DIM])
discount_in = tf.placeholder(tf.float32)
row_indices_in = tf.placeholder(tf.int32, [None])
not_done_in = tf.placeholder(tf.float32, [None])

#copy over the current network to the target network
a1 = w3.assign(w1)
a2 = w4.assign(w2)

#table of action indices
actions_indices = tf.stack([row_indices_in, a_in],axis=1)

#constants (inferred or otherwise)
batch_size = tf.shape(s_in)[0]

q_out = tf.matmul(tf.nn.relu(tf.matmul(s_in,w1)), w2)  #original network
q1_out = tf.matmul(tf.nn.relu(tf.matmul(s_in,w3)), w4) #target network

target = r_in + discount_in * not_done_in * tf.stop_gradient(tf.reduce_max(q1_out,axis=1))

bellman_residual = target - tf.gather_nd(q_out,actions_indices) #q_out[0][a_in[0]] #

thetas = [item for item in tf.trainable_variables()]
reg_losses = [LAMBDA * tf.nn.l2_loss(item) for item in tf.trainable_variables() if 'weight' in item.name]

loss = 0.5*tf.reduce_mean(tf.square(bellman_residual)) + tf.reduce_sum(reg_losses)
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)



def run():
	assignOps = [a1,a2]

	losses = np.zeros([NUM_TRIALS,NUM_EPISODES])
	bellman_losses = np.zeros([NUM_TRIALS,NUM_EPISODES])
	disc_rewards = np.zeros([NUM_TRIALS,NUM_EPISODES])
	aver_moves = np.zeros([NUM_TRIALS,NUM_EPISODES])

	for trial in range(NUM_TRIALS):
		S_BUFF = np.zeros([BUFFER_SIZE,STATE_DIM])
		A_BUFF = np.zeros([BUFFER_SIZE,SCALAR_DIM])
		R_BUFF = np.zeros([BUFFER_SIZE,SCALAR_DIM])
		S1_BUFF = np.zeros([BUFFER_SIZE,STATE_DIM])
		ND_BUFF = np.zeros([BUFFER_SIZE,SCALAR_DIM])  #not-done buffer

		init_op = tf.global_variables_initializer()
		saver = tf.train.Saver(write_version = tf.train.SaverDef.V1)
		with tf.Session() as sess:
			sess.run(init_op)	# init vars
			step = -1
			for episode in range(NUM_EPISODES):
				if episode%MODULO==0:
					print("Updating target network...")
					for op in assignOps:
						_ = sess.run(op)
					# z1,z2,z3,z4 = sess.run([w1,w2,w3,w4])
					# print(np.mean((z1==z3)*1.0))
					# print(np.mean((z2==z4)*1.0))

				s_t = env.reset()
				l1 = 0.0
				bellman_l1 = 0.0
				ep_steps = 0.0
				#1) first we experience:
				while 1:
					#select action
					if np.random.random()>EPSILON:
						qs = sess.run(q_out,feed_dict={s_in:np.expand_dims(s_t,axis=0)})
						a_t = np.argmax(qs)				
					else:
						a_t = np.random.choice(ACTION_DIM)
					step+=1
					ep_steps+=1

					#take a step in the env 
					s1_t, r_t, done, info = env.step(a_t)
					s1_t, r_t, done, info = modify_outputs(s1_t, r_t, done, info)
					not_done = (not done)*1.0
					S_BUFF[step%BUFFER_SIZE] = s_t
					A_BUFF[step%BUFFER_SIZE] = a_t
					R_BUFF[step%BUFFER_SIZE] = r_t
					S1_BUFF[step%BUFFER_SIZE] = s1_t
					ND_BUFF[step%BUFFER_SIZE] = not_done

					#2) second we train:
					inds = np.random.choice(len(S_BUFF),MINI_BATCH_SIZE)
					S = S_BUFF[inds]
					A = np.reshape(A_BUFF[inds], MINI_BATCH_SIZE)
					R = np.reshape(R_BUFF[inds], MINI_BATCH_SIZE)
					S1 = S1_BUFF[inds]
					ND = np.reshape(ND_BUFF[inds], MINI_BATCH_SIZE)
				
					[l2,bellman_l2,_] = sess.run([loss,bellman_residual,train_op], feed_dict={
									s_in:S,
									a_in:A,
									r_in:R,
									s1_in:S1,
									discount_in:DISCOUNT,
									row_indices_in:np.arange(len(S)),
									not_done_in:ND
									})
					l1 += l2
					bellman_l1 += bellman_l2

					if done == True:
						break
					s_t = s1_t


				#3) third we evaluate:
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
				if episode%1==0:
					print("Trial:",trial,"Episode", episode,"loss:",'{0:.4f}'.format(l1),"steps:",int(ep_steps),"\tAverage performance over",NUM_EPISODES_EVAL,"evaluation trials: Moves",'{0:.3f}'.format(ave),"| Reward",'{0:.4f}'.format(rew))
				losses[trial,episode] = l1/ep_steps
				bellman_losses[trial,episode] = np.mean(bellman_l1)/ep_steps
				disc_rewards[trial,episode] = rew
				aver_moves[trial,episode] = ave
			saver.save(sess,MODEL_FILENAME)
			print("Saved model at",MODEL_FILENAME)	

	# print("losses",losses)
	# print("bellman", bellman_losses)
	# print("disc_rew", disc_rewards)
	# print("aver_moves",aver_moves)

	concat = np.concatenate([losses,bellman_losses,disc_rewards,aver_moves])
	# print(concat)

	print(SAVE_FILENAME)
	with open(SAVE_FILENAME,'wb') as f:
		np.savetxt(SAVE_FILENAME, concat, fmt='%.5f',delimiter=",")



if __name__ == '__main__':
	run()








# def run():
# 	losses = np.zeros([NUM_TRIALS,NUM_EPISODES])
# 	bellman_losses = np.zeros([NUM_TRIALS,NUM_EPISODES])
# 	disc_rewards = np.zeros([NUM_TRIALS,NUM_EPISODES])
# 	aver_moves = np.zeros([NUM_TRIALS,NUM_EPISODES])

# 	for trial in range(NUM_TRIALS):
# 		init_op = tf.global_variables_initializer()
# 		saver = tf.train.Saver(write_version = tf.train.SaverDef.V1)
# 		with tf.Session() as sess:
# 			sess.run(init_op)	# init vars
# 			for episode in range(NUM_EPISODES):
# 				s_t = env.reset()
# 				l1 = 0.0
# 				bellman_l1 = 0.0
# 				steps = 0
# 				#first we train:
# 				while 1:
# 					#select action
# 					if np.random.random()>EPSILON:
# 						qs = sess.run(q_out,feed_dict={s_in:np.expand_dims(s_t,axis=0)})
# 						a_t = np.argmax(qs)				
# 					else:
# 						a_t = np.random.choice(ACTION_DIM)
# 					steps+=1
					
# 					#take a step in the env 
# 					s_t1, r_t1, done, info = env.step(a_t)
# 					s_t1, r_t1, done, info = modify_outputs(s_t1, r_t1, done, info)
# 					not_done = not done

# 					#learn
# 					[l2,bellman_l2,_] = sess.run([loss,bellman_residual,train_op], feed_dict={
# 									s_in:np.array([s_t]),
# 									a_in:np.array([a_t]),
# 									r_in:np.array([r_t1]),
# 									s1_in:np.array([s_t1]),
# 									discount_in:DISCOUNT,
# 									row_indices_in:np.arange(1),
# 									not_done_in:np.array([not_done*1]),
# 									current_iter:int(episode),
# 									modulo:int(MODULO)
# 									})
# 					l1 += l2
# 					bellman_l1 += bellman_l2
# 					if done == True:
# 						break
# 					s_t = s_t1


# 				#then we evaluate:
# 				ave = 0
# 				time_per_episode = np.zeros(NUM_EPISODES_EVAL)
# 				reward_per_episode = np.zeros(NUM_EPISODES_EVAL)
# 				for episode_eval in range(NUM_EPISODES_EVAL):
# 					s_t = env.reset()
# 					episode_reward = 0.0
# 					cumulative_discount = 1.0
# 					for t in range(MAX_EPISODE_LEN):
# 						qs = sess.run(q_out,feed_dict={s_in:np.expand_dims(s_t,axis=0)})
# 						a_t = np.argmax(qs)
# 						# print(qs,a_t)
# 						s_t, r_t1, done, info = env.step(a_t)
# 						s_t, r_t1, done, info = modify_outputs(s_t, r_t1, done, info)
# 						episode_reward += cumulative_discount*r_t1
# 						cumulative_discount = cumulative_discount * DISCOUNT
# 						if info != {}: print(info)
# 						if done:
# 							break
# 					# print(t)
# 					time_per_episode[episode_eval] = t+1
# 					reward_per_episode[episode_eval] = episode_reward
# 				ave = np.mean(time_per_episode)
# 				rew = np.mean(reward_per_episode)
# 				l1 = l1/steps
# 				bellman_l1 = bellman_l1/steps
# 				if episode%1==0:
# 					print("Trial:",trial,"Episode", episode,"loss:",'{0:.4f}'.format(l1),"\tAverage performance over",NUM_EPISODES_EVAL,"evaluation trials: Moves",'{0:.3f}'.format(ave),"| Reward",'{0:.4f}'.format(rew))
# 				losses[trial,episode] = l1
# 				bellman_losses[trial,episode] = bellman_l1
# 				disc_rewards[trial,episode] = rew
# 				aver_moves[trial,episode] = ave
# 			saver.save(sess,MODEL_FILENAME)
# 			print("Saved model at",MODEL_FILENAME)	

# 	# print("losses",losses)
# 	# print("bellman", bellman_losses)
# 	# print("disc_rew", disc_rewards)
# 	# print("aver_moves",aver_moves)

# 	concat = np.concatenate([losses,bellman_losses,disc_rewards,aver_moves])
# 	# print(concat)

# 	print(SAVE_FILENAME)
# 	with open(SAVE_FILENAME,'wb') as f:
# 		np.savetxt(SAVE_FILENAME, concat,delimiter=",")
