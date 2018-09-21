#Problem A, task 3a
import sys
import numpy as np
import gym
import tensorflow as tf

np.random.seed(42) #for repeatability

#initialise environment
env = gym.make('CartPole-v0')

print(env.action_space)
z = env.action_space
a = z.n
print(a)

