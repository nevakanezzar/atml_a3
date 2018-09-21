#Problem B, task 4 -- test
from loadmodels import LoadModels
import sys
import numpy as np
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
import random
import time
from copy import deepcopy
from PIL import Image
import select


try:
	GAME_INDEX = int(sys.argv[1])
	if GAME_INDEX == 2:
		STATE_X = 40 
		STATE_Y = 40
	else:
		STATE_X = 60 
		STATE_Y = 60
	LEARNING_RATE = 0.0
except:
	print("Something went wrong. Signature is game_index")


#arguments


#constants
GAMES = ['Pong-v3','MsPacman-v3','Boxing-v3']
DISCOUNT = 0.99
EPSILON = 1
FRAMES = 4
MODULO = 5000  # TO DO 5000
NUM_STEPS = 2000000  # TO DO 1000000
NUM_EPISODES_EVAL = 100 # TO DO 100
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
# STATE_X = 60 
# STATE_Y = 60
SCALAR_DIM = 1
ACTION_DIM = env.action_space.n
print('Game : '+game)
print('Possible actions : {}'.format(ACTION_DIM))

#filenames
MODEL_FOLDER = '../model/'
SAVE_FOLDER = MODEL_FOLDER
TENSORBOARD_FOLDER = '.'

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
	#creates a downsampled greyscale image for Pong
	if GAME_INDEX == 0:
		in_img = np.ones(input_img.shape, dtype=np.uint8) * 255
		in_img[input_img == 144] = 0
		in_img[input_img == 72] = 0
		in_img[input_img == 17] = 0
		input_img = deepcopy(in_img)

	img = Image.fromarray(input_img[X1:X2,Y1:Y2], 'RGB').convert('L')
	img = img.resize((STATE_X, STATE_Y), Image.ANTIALIAS)
	
	output_img = np.array(img).astype(np.uint8)
	
	return output_img


#hyperparameters
# LEARNING_RATE = 0.01
BUFFER_SIZE = 400000  #TO DO 100000
MINI_BATCH_SIZE = 32
LAMBDA = 0.0
STD = 0.1


#file names
# year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
# MODEL = "b34_"+sys.argv[1]+"_"+sys.argv[2]+"_"+sys.argv[3]+"_"+hour+'_'+minute
# MODEL_FILENAME = MODEL_FOLDER+MODEL+".model"
# TRAIN_SAVE_FILENAME = SAVE_FOLDER+MODEL+"train.csv"
# TEST_SAVE_FILENAME = SAVE_FOLDER+MODEL+"test.csv"


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
	s_in1 = tf.transpose(s_in, [0,3,2,1])
	s1_in1 = tf.transpose(s1_in, [0,3,2,1])

	c1_height = 6
	c1_width = 6
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
	W1 = tf.get_variable("weight1", shape=c1_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
	W2 = tf.get_variable("weight2", shape=c2_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
	W3 = tf.get_variable("weight3", shape=[FC_IN_DIM, FC_OUT_DIM], initializer=tf.contrib.layers.xavier_initializer())
	W4 = tf.get_variable("weight4", shape=[FC_OUT_DIM, ACTION_DIM], initializer=tf.contrib.layers.xavier_initializer())
	B1 = tf.get_variable("bias1", shape=[c1_shape[-1]], initializer=tf.constant_initializer(0.0))
	B2 = tf.get_variable("bias2", shape=[c2_shape[-1]], initializer=tf.constant_initializer(0.0))
	B3 = tf.get_variable("bias3", shape=[FC_OUT_DIM], initializer=tf.constant_initializer(0.0))
	B4 = tf.get_variable("bias4", shape=[ACTION_DIM], initializer=tf.constant_initializer(0.0))

	#q_s1a1 network variables
	W5 = tf.get_variable("weight5", shape=c1_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
	W6 = tf.get_variable("weight6", shape=c2_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
	W7 = tf.get_variable("weight7", shape=[FC_IN_DIM, FC_OUT_DIM], initializer=tf.contrib.layers.xavier_initializer())
	W8 = tf.get_variable("weight8", shape=[FC_OUT_DIM, ACTION_DIM], initializer=tf.contrib.layers.xavier_initializer())
	B5 = tf.get_variable("bias5", shape=[c1_shape[-1]], initializer=tf.constant_initializer(0.0))
	B6 = tf.get_variable("bias6", shape=[c2_shape[-1]], initializer=tf.constant_initializer(0.0))
	B7 = tf.get_variable("bias7", shape=[FC_OUT_DIM], initializer=tf.constant_initializer(0.0))
	B8 = tf.get_variable("bias8", shape=[ACTION_DIM], initializer=tf.constant_initializer(0.0))


	#q_sa network 
	conv1 = tf.nn.relu(tf.nn.conv2d(s_in1, W1, strides=[1,2,2,1], padding="SAME") + B1)
	conv1 = tf.nn.relu(tf.nn.conv2d(conv1, W2, strides=[1,2,2,1], padding="SAME") + B2)
	conv1 = tf.nn.relu(tf.matmul(tf.reshape(conv1, [batch_size,-1]), W3) + B3)
	q_out = tf.matmul(conv1,W4) + B4

	#q_s1a1 network 
	conv2 = tf.nn.relu(tf.nn.conv2d(s1_in1, W5, strides=[1,2,2,1], padding="SAME") + B5)
	conv2 = tf.nn.relu(tf.nn.conv2d(conv2,  W6, strides=[1,2,2,1], padding="SAME") + B6)
	conv2 = tf.nn.relu(tf.matmul(tf.reshape(conv2, [batch_size,-1]), W7) + B7)
	q1_out = tf.matmul(conv2,W8) + B8


	target = r_in + discount_in * not_done_in * tf.stop_gradient(tf.reduce_max(q1_out,axis=1))

	bellman_residual = target - tf.gather_nd(q_out,actions_indices)

	thetas = [item for item in tf.trainable_variables()]
	reg_losses = [LAMBDA * tf.nn.l2_loss(item) for item in tf.trainable_variables() if 'weight' in item.name]

	loss = 0.5*tf.reduce_mean(tf.square(bellman_residual)) #+ tf.reduce_sum(reg_losses)
	train_op = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss)

	#copy the network over, if it's time
	# a1 = W5.assign(W1)
	# a2 = W6.assign(W2)
	# a3 = W7.assign(W3)
	# a4 = W8.assign(W4)
	# a5 = B5.assign(B1)
	# a6 = B6.assign(B2)
	# a7 = B7.assign(B3)
	# a8 = B8.assign(B4)


def load(LOAD_FILENAME):
	print("\n\n\n\nEvaluating model:",LOAD_FILENAME)
	inp2 = ''
	EVAL_TRIALS = 100
	eS = np.zeros([4, STATE_X, STATE_Y],dtype=np.uint8)
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		saver = tf.train.Saver(write_version = tf.train.SaverDef.V1)  
		saver.restore(sess,LOAD_FILENAME)
		ave = 0
		time_per_episode = np.zeros(EVAL_TRIALS)
		reward_per_episode = np.zeros(EVAL_TRIALS)
		for episode_eval in range(NUM_EPISODES_EVAL):
			e_ep_steps = 0
			e_ep_disc_rew = 0.0
			e_s_t = env.reset()	
			eS[0] = eS[1] = eS[2] = eS[3] = preprocess_state(e_s_t)
			while 1:
				inp,_,_ = select.select([sys.stdin],[],[],0)
				for s in inp:
					if s == sys.stdin:
						inp2 = sys.stdin.readline()
						inp2 = inp2[:-1].lower()
				qs = sess.run(q_out,feed_dict={s_in:[eS]})
				e_a_t = np.argmax(qs)				
				e_s1_t, e_r_t, e_done, e_info = env.step(e_a_t)
				e_s1_t, e_r_t, e_not_done, e_info = modify_outputs(e_s1_t, e_r_t, e_done, e_info)
				if inp2=='r': 
					env.render()
				else:
					env.render(close=True)
				e_ep_disc_rew += e_r_t * DISCOUNT**e_ep_steps
				e_ep_steps += 1
				if e_done:
					break
				else:
					eS[0:3]=eS[1:4]
					eS[3] = preprocess_state(e_s1_t)

			time_per_episode[episode_eval] = e_ep_steps+1
			reward_per_episode[episode_eval] = e_ep_disc_rew
			print("Trial:",episode_eval,"\tMoves",'{0:.3f}'.format(e_ep_steps+1),"| Reward",'{0:.4f}'.format(e_ep_disc_rew))

		ave = np.mean(time_per_episode)
		rew = np.mean(reward_per_episode)
		print("Average performance over",EVAL_TRIALS,"evaluation trials: Moves",'{0:.3f}'.format(ave),"| Reward",'{0:.4f}'.format(rew))



if __name__ == '__main__':
	try:
		stub = 'b34_'+sys.argv[1]
	except:
		print("Something went wrong. Try adding a game number (0,1,2)")
	models = LoadModels(stub)	
	print("\nYou are about to run 100 evaluations of these trained models.")
	print("At any time during the evaluation, you may type 'r' and hit return to see the environment being rendered.")
	print("Enter any other key besides 'r' to stop the rendering.")
	x = input("Press Enter to continue...")
	for model in models:
		load(MODEL_FOLDER+model)

