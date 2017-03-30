#Problem A, task 1 and 2

import numpy as np
import gym
env = gym.make('CartPole-v0')


#constants
DISCOUNT = 0.99
NUM_EPS = 100
MAX_EP_LEN = 300

# task 1: generate 3 trajectories under a uniform random policy. report episode lengths and return
print("\n\n\nProblem A: 1")
for episode in range(3):
	observation = env.reset()
	episode_reward = 0.0
	cumulative_discount = DISCOUNT
	for t in range(MAX_EP_LEN):
		#env.render()
		action = env.action_space.sample()
		print(observation, action)
		observation, _, done, info = env.step(action)
		if done == True:
			reward = -1
		else:
			reward = 0
		episode_reward += cumulative_discount*reward #modifies reward to 0 on non-terminating steps and -1 on termination
		cumulative_discount = cumulative_discount * DISCOUNT
		if done:
			print("Episode finished after {} timesteps".format(t+1),"\tReward from initial state is {}".format(episode_reward))
			break


# task 2: generate 100 episodes under a uniform random policy. report mean and std of episode lengths and return
print("\n\n\nProblem A: 2")
time_per_episode = np.zeros(NUM_EPS)
reward_per_episode = np.zeros(NUM_EPS)
for episode in range(NUM_EPS):
	observation = env.reset()
	episode_reward = 0.0
	cumulative_discount = DISCOUNT
	for t in range(100):
		action = env.action_space.sample()
		observation, _, done, info = env.step(action)
		if done == True:
			reward = -1
		else:
			reward = 0
		episode_reward += cumulative_discount*reward #modifies reward to 0 on non-terminating steps and -1 on termination
		cumulative_discount = cumulative_discount * DISCOUNT
		if done:
			print("Episode finished after {} timesteps".format(t+1),"\tReward from initial state is {}".format(episode_reward))
			time_per_episode[episode] = t+1
			reward_per_episode[episode] = episode_reward
			break

print("Mean and std. of episode times:", np.mean(time_per_episode), np.std(time_per_episode))
print("Mean and std. of episode rewards:", np.mean(reward_per_episode), np.std(reward_per_episode))







