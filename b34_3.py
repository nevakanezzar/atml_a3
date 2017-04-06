#Problem B, task 3 and 4
import sys
import numpy as np
# import matplotlib.pyplot as plt
import gym
import tensorflow as tf
import random
import time
from PIL import Image
import select


try:
	GAME_INDEX = int(sys.argv[1])
	LEARNING_RATE = float(sys.argv[2])
	DISCOUNT_STARTING_FACTOR = int(sys.argv[3])
except:
	print("Something went wrong. Signature is game_index learning_rate discount_starting_factor")


#arguments
# GAME_INDEX = 2

#constants
GAMES = ['Pong-v3','MsPacman-v3','Boxing-v3']
DISCOUNT = 0.99
EPSILON = 1
FRAMES = 4
MODULO = 5000  # TO DO 5000
NUM_STEPS = 1000000  # TO DO 1000000
NUM_EPISODES_EVAL = 10 # TO DO 100
EVAL_EVERY = 50000 #TO DO 50000

#set (x1,y1)-(x2,y2) constraints for each game
if GAME_INDEX == 0:
	X1 = 30
	Y1 = 0
	X2 = 200
	Y2 = 160
elif GAME_INDEX == 1:
	X1 = 0
	Y1 = 0
	X2 = 180
	Y2 = 160
elif GAME_INDEX == 2:
	X1 = 30
	Y1 = 30
	X2 = 180
	Y2 = 130


#initialise environment
game = GAMES[GAME_INDEX]
env = gym.make(game)
STATE_DIM = [210,160,3]
STATE_X = 40 
STATE_Y = 40
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
	not_ter = int((not ter) * 1.0)
	return obs, int(rew), not_ter, inf


#function to preprocess states into useable_states 
def preprocess_state(input_img):
	#creates a downsampled greyscale image
	
	img = Image.fromarray(input_img[X1:X2,Y1:Y2], 'RGB').convert('L')
	img = img.resize((STATE_X, STATE_Y), Image.ANTIALIAS)
	
	output_img = np.array(img).astype(np.uint8)
	
	# plt.imshow(output_img)
	# plt.show()
	# plt.close()

	return output_img


#hyperparameters
# LEARNING_RATE = 0.01
BUFFER_SIZE = 100000  #TO DO 100000
MINI_BATCH_SIZE = 32
LAMBDA = 0.0


#file names
year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
MODEL = "b34_"+sys.argv[1]+"_"+sys.argv[2]+"_"+sys.argv[3]+"_"+hour+'_'+minute
MODEL_FILENAME = MODEL_FOLDER+MODEL+".model"
TRAIN_SAVE_FILENAME = SAVE_FOLDER+MODEL+"train.csv"
TEST_SAVE_FILENAME = SAVE_FOLDER+MODEL+"test.csv"


tf.reset_default_graph()
#create q learning graph

#tf functions

with tf.device('/gpu:0'):
	#prediction inputs
	s_in = tf.placeholder(tf.float32, [None,FRAMES, STATE_X, STATE_Y])
	a_in = tf.placeholder(tf.int32, [None])
	r_in = tf.placeholder(tf.float32, [None])
	s1_in = tf.placeholder(tf.float32, [None,FRAMES, STATE_X, STATE_Y])
	discount_in = tf.placeholder(tf.float32)
	row_indices_in = tf.placeholder(tf.int32, [None])
	not_done_in = tf.placeholder(tf.float32, [None])

	#table of action indices
	actions_indices = tf.stack([row_indices_in, a_in],axis=1)

	#constants (inferred or otherwise)
	batch_size = tf.shape(s_in)[0]

	#reshapes for conv net
	s_in1 = tf.transpose(s_in, [0,2,3,1])
	s1_in1 = tf.transpose(s1_in, [0,2,3,1])

	c1_height = 8
	c1_width = 8
	c1_in_channels = FRAMES
	c1_out_channels = 16
	c1_shape = [c1_height, c1_width, c1_in_channels, c1_out_channels]

	c2_height = 4
	c2_width = 4
	c2_in_channels = 16
	c2_out_channels = 32
	c2_shape = [c2_height, c2_width, c2_in_channels, c2_out_channels]

	FC_IN_DIM = STATE_X * STATE_Y * c2_out_channels / 16
	FC_OUT_DIM = 256

	#q_sa network variables
	W1 = tf.get_variable("weight1", shape=c1_shape, initializer=tf.contrib.layers.xavier_initializer())
	W2 = tf.get_variable("weight2", shape=c2_shape, initializer=tf.contrib.layers.xavier_initializer())
	W3 = tf.get_variable("weight3", shape=[FC_IN_DIM, FC_OUT_DIM], initializer=tf.contrib.layers.xavier_initializer())
	W4 = tf.get_variable("weight4", shape=[FC_OUT_DIM, ACTION_DIM], initializer=tf.contrib.layers.xavier_initializer())
	B1 = tf.get_variable("bias1", shape=[c1_shape[-1]], initializer=tf.contrib.layers.xavier_initializer())
	B2 = tf.get_variable("bias2", shape=[c2_shape[-1]], initializer=tf.contrib.layers.xavier_initializer())
	B3 = tf.get_variable("bias3", shape=[FC_OUT_DIM], initializer=tf.contrib.layers.xavier_initializer())
	B4 = tf.get_variable("bias4", shape=[ACTION_DIM], initializer=tf.contrib.layers.xavier_initializer())

	#q_s1a1 network variables
	W5 = tf.get_variable("weight5", shape=c1_shape, initializer=tf.contrib.layers.xavier_initializer())
	W6 = tf.get_variable("weight6", shape=c2_shape, initializer=tf.contrib.layers.xavier_initializer())
	W7 = tf.get_variable("weight7", shape=[FC_IN_DIM, FC_OUT_DIM], initializer=tf.contrib.layers.xavier_initializer())
	W8 = tf.get_variable("weight8", shape=[FC_OUT_DIM, ACTION_DIM], initializer=tf.contrib.layers.xavier_initializer())
	B5 = tf.get_variable("bias5", shape=[c1_shape[-1]], initializer=tf.contrib.layers.xavier_initializer())
	B6 = tf.get_variable("bias6", shape=[c2_shape[-1]], initializer=tf.contrib.layers.xavier_initializer())
	B7 = tf.get_variable("bias7", shape=[FC_OUT_DIM], initializer=tf.contrib.layers.xavier_initializer())
	B8 = tf.get_variable("bias8", shape=[ACTION_DIM], initializer=tf.contrib.layers.xavier_initializer())

	#copy the network over, if it's time

	a1 = W5.assign(W1)
	a2 = W6.assign(W2)
	a3 = W7.assign(W3)
	a4 = W8.assign(W4)
	a5 = B5.assign(B1)
	a6 = B6.assign(B2)
	a7 = B7.assign(B3)
	a8 = B8.assign(B4)


	#q_sa network 
	conv1 = tf.nn.relu(tf.nn.conv2d(s_in1, W1, strides=[1,2,2,1], padding="SAME") + B1)
	conv1 = tf.nn.relu(tf.nn.conv2d(conv1, W2, strides=[1,2,2,1], padding="SAME") + B2)
	conv1 = tf.nn.relu(tf.matmul(tf.reshape(conv1, [batch_size,-1]), W3) + B3)
	q_out = tf.matmul(conv1,W4) + B4

	#q_s1a1 network 
	conv2 = tf.nn.relu(tf.nn.conv2d(s1_in1, W5, strides=[1,2,2,1], padding="SAME") + B5)
	conv2 = tf.nn.relu(tf.nn.conv2d(conv2,  W6, strides=[1,2,2,1], padding="SAME") + B6)
	conv2 = tf.nn.relu(tf.matmul(tf.reshape(conv1, [batch_size,-1]), W7) + B7)
	q1_out = tf.matmul(conv1,W8) + B8


	target = r_in + discount_in * not_done_in * tf.stop_gradient(tf.reduce_max(q1_out,axis=1))

	bellman_residual = target - tf.gather_nd(q_out,actions_indices)

	thetas = [item for item in tf.trainable_variables()]
	reg_losses = [LAMBDA * tf.nn.l2_loss(item) for item in tf.trainable_variables() if 'weight' in item.name]

	loss = 0.5*tf.reduce_mean(tf.square(bellman_residual)) + tf.reduce_sum(reg_losses)
	train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)


def run():
	e_avg_disc_rew_BEST = -99999999999999.9999999999

	assignOps = [a1,a2,a3,a4,a5,a6,a7,a8]
	S = np.zeros([BUFFER_SIZE+2*FRAMES, STATE_X, STATE_Y],dtype=np.uint8)
	A = np.zeros([BUFFER_SIZE+2*FRAMES, SCALAR_DIM],dtype=np.uint8)
	R = np.zeros([BUFFER_SIZE+2*FRAMES, SCALAR_DIM],dtype=np.int8)
	ND = np.zeros([BUFFER_SIZE+2*FRAMES, SCALAR_DIM],dtype=np.uint8)
	
	train_losses = []
	train_bellman = []
	test_steps = np.zeros([NUM_STEPS//EVAL_EVERY+2,NUM_EPISODES_EVAL])
	test_disc_rew = np.zeros([NUM_STEPS//EVAL_EVERY+2,NUM_EPISODES_EVAL])
	
	next_eval = EVAL_EVERY
	eS = np.zeros([4, STATE_X, STATE_Y],dtype=np.uint8)

	init_op = tf.global_variables_initializer()
	saver = tf.train.Saver(write_version = tf.train.SaverDef.V1)
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

		sess.run(init_op)

		input = ""
		ep_steps = 0
		l1 = 0.0
		bellman_l1 = 0.0
		disc_rew = tot_rew = 0.0
		s_t = env.reset()	
		S[0] = S[1] = S[2] = S[3] = preprocess_state(s_t)

		for steps in range(FRAMES-1,NUM_STEPS+FRAMES-1):
			
			inp,_,_ = select.select([sys.stdin],[],[],0)
			for s in inp:
				if s == sys.stdin:
					input = sys.stdin.readline()
					input = input[:-1].lower()
			
			EPSILON = (0.9 * 0.99999**(DISCOUNT_STARTING_FACTOR+steps)) + 0.1
			if np.random.random()>EPSILON:
				S_in = np.take(S,[range((steps%BUFFER_SIZE)-FRAMES+1,(steps%BUFFER_SIZE)+1)], axis=0)
				qs = sess.run(q_out,feed_dict={s_in:S_in})
				a_t = np.argmax(qs)				
			else:
				a_t = np.random.choice(ACTION_DIM)
			s1_t, r_t, done, info = env.step(a_t)
			s1_t, r_t, not_done, info = modify_outputs(s1_t, r_t, done, info)
			if input=='r': 
				env.render()
			else:
				env.render(close=True)

			tot_rew += r_t
			disc_rew += r_t * DISCOUNT**ep_steps

			A[steps%BUFFER_SIZE] = a_t
			R[steps%BUFFER_SIZE] = r_t
			S[(steps+1)%BUFFER_SIZE] = preprocess_state(s1_t)
			ND[steps%BUFFER_SIZE] = not_done

			ind_starts = np.random.choice(min(steps+1,BUFFER_SIZE)-FRAMES+1,MINI_BATCH_SIZE)
			ind_ends = ind_starts + FRAMES
			inds_S = [range(i,i+FRAMES) for i in ind_starts]
			inds_S1 = [range(i+1,i+1+FRAMES) for i in ind_starts]

			S_in = np.take(S,inds_S, axis=0)
			A_in = np.take(A,ind_ends)
			R_in = np.take(R,ind_ends)
			S1_in = np.take(S,inds_S1, axis=0)
			ND_in = np.take(ND,ind_ends)

			if steps%MODULO==0:
				# print("Updating target network...")
				for op in assignOps:
					_ = sess.run(op)

			[l2,bellman_l2,_] = sess.run([loss,bellman_residual,train_op], feed_dict={
									s_in:S_in,
									a_in:A_in,
									r_in:R_in,
									s1_in:S1_in,
									discount_in:DISCOUNT,
									row_indices_in:np.arange(len(S_in)),
									not_done_in:ND_in
									})
			l1 += l2
			bellman_l1 += np.mean(bellman_l2)

			if done:
				train_losses.append(l1/ep_steps)
				train_bellman.append(bellman_l1/ep_steps)
				print("Steps:",steps,"\tEpisode steps:",ep_steps,"\tTot rew:",tot_rew,"\tDisc rew:",disc_rew,"\tLoss:",l1/ep_steps,"\tBellman:",bellman_l1/ep_steps, flush=True)
				
				if steps > next_eval:  #if more than evaluation checkpoint, eval
					next_eval += EVAL_EVERY
					i = steps//EVAL_EVERY
					e_tot_rew = 0
					for j in range(NUM_EPISODES_EVAL):
						e_ep_steps = 0
						e_ep_disc_rew = 0.0
						e_s_t = env.reset()	
						eS[0] = eS[1] = eS[2] = eS[3] = preprocess_state(e_s_t)
						while 1:
							inp,_,_ = select.select([sys.stdin],[],[],0)
							for s in inp:
								if s == sys.stdin:
									input = sys.stdin.readline()
									input = input[:-1].lower()
							qs = sess.run(q_out,feed_dict={s_in:[eS]})
							e_a_t = np.argmax(qs)				
							e_s1_t, e_r_t, e_done, e_info = env.step(e_a_t)
							e_s1_t, e_r_t, e_not_done, e_info = modify_outputs(e_s1_t, e_r_t, e_done, e_info)
							if input=='r': 
								env.render()
							else:
								env.render(close=True)
							e_tot_rew += e_r_t
							e_ep_disc_rew += e_r_t * DISCOUNT**e_ep_steps
							e_ep_steps += 1
							if e_done:
								break
							else:
								eS[0:3]=eS[1:4]
								eS[3] = preprocess_state(e_s1_t)

						test_steps[i,j] = e_ep_steps
						test_disc_rew[i,j] = e_ep_disc_rew

					saver.save(sess,MODEL_FILENAME+"_"+str(i))
					e_avg_steps = np.mean(test_steps[i,:])
					e_avg_disc_rew = np.mean(test_disc_rew[i,:])
					e_avg_tot_rew = float(e_tot_rew)/NUM_EPISODES_EVAL
					print("Evaluation over",NUM_EPISODES_EVAL,"episodes at",steps,"steps | Avg steps:",e_avg_steps,"Avg tot rew:",e_avg_tot_rew,"Avg disc rew:",e_avg_disc_rew)
					if e_avg_disc_rew > e_avg_disc_rew_BEST:
						e_avg_disc_rew_BEST = e_avg_disc_rew 
						saver.save(sess,MODEL_FILENAME)
						print("Saved model at",MODEL_FILENAME)


				ep_steps = 0
				l1 = 0.0
				bellman_l1 = 0.0
				disc_rew = tot_rew = 0.0
				s_t = env.reset()
				S[(steps+1)%BUFFER_SIZE] = preprocess_state(s_t)

			else:
				ep_steps += 1
				s_t = s1_t

		i = int(steps//EVAL_EVERY + 1)
		e_tot_rew = 0
		for j in range(NUM_EPISODES_EVAL):
			e_ep_steps = 0
			e_ep_disc_rew = 0.0
			e_s_t = env.reset()	
			eS[0] = eS[1] = eS[2] = eS[3] = preprocess_state(e_s_t)
			while 1:
				inp,_,_ = select.select([sys.stdin],[],[],0)
				for s in inp:
					if s == sys.stdin:
						input = sys.stdin.readline()
						input = input[:-1].lower()
				qs = sess.run(q_out,feed_dict={s_in:[eS]})
				e_a_t = np.argmax(qs)				
				e_s1_t, e_r_t, e_done, e_info = env.step(e_a_t)
				e_s1_t, e_r_t, e_not_done, e_info = modify_outputs(e_s1_t, e_r_t, e_done, e_info)
				if input=='r': 
					env.render()
				else:
					env.render(close=True)
				e_tot_rew += e_r_t
				e_ep_disc_rew += e_r_t * DISCOUNT**e_ep_steps
				e_ep_steps += 1
				if e_done:
					break
				else:
					eS[0:3]=eS[1:4]
					eS[3] = preprocess_state(e_s1_t)

			test_steps[i,j] = e_ep_steps
			test_disc_rew[i,j] = e_ep_disc_rew

		e_avg_steps = np.mean(test_steps[i,:])
		e_avg_disc_rew = np.mean(test_disc_rew[i,:])
		e_avg_tot_rew = float(e_tot_rew)/NUM_EPISODES_EVAL
		print("Evaluation over",NUM_EPISODES_EVAL,"episodes at",steps,"steps | Avg steps:",e_avg_steps,"Avg tot rew:",e_avg_tot_rew,"Avg disc rew:",e_avg_disc_rew)
		if e_avg_disc_rew > e_avg_disc_rew_BEST:
			e_avg_disc_rew_BEST = e_avg_disc_rew 
			saver.save(sess,MODEL_FILENAME)
			print("Saved model at",MODEL_FILENAME)

		# saver.save(sess,MODEL_FILENAME)
		# print("Saved model at",MODEL_FILENAME)

	concat = np.concatenate([train_losses,train_bellman])
	with open(TRAIN_SAVE_FILENAME,'wb') as f:
		np.savetxt(TRAIN_SAVE_FILENAME, concat, fmt='%.5f',delimiter=",")
	print("Saved training stats to:",TRAIN_SAVE_FILENAME)

	concat = np.concatenate([test_steps,test_disc_rew])
	with open(TEST_SAVE_FILENAME,'wb') as f:
		np.savetxt(TEST_SAVE_FILENAME, concat, fmt='%.5f',delimiter=",")
	print("Saved testing stats to:",TEST_SAVE_FILENAME)


	return None
	

run()

