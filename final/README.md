README.md 

Name: Sudhanshu Kasewa
Student Number: 15115014
Course: COMPGI13 - Advanced Topics in Machine Learning
Coursework: Assignment 3

This readme file details the structure of the Assignment 3 submission as well as the commands for running the code.


1. Structure:

The submission is structured as follows:

a) 15115014_assignment3.zip -- Parent zip file, containing the following:
b) code -- folder containing code.
c) model -- folder containing trained models. Folder contains individual files for models.
d) 15115014_report.pdf -- a pdf report of the experiments carried out for the assignment.
e) README.md -- this readme file.

The submission expects:
i) python 3.5+ or compatible
ii) tensorflow
iii) gym with cartpole and atari


2. Python and Tensorflow versions

This assignment was completed using python 3.5, and tensorflow version 1.0.


3. Running the code

The code must be run from the code folder. The code can be run with the following commands:

I. PROBLEM A:

a) Training:
	i)	python3 a12.py 
	ii) 	python3 a3a.py LEARNING_RATE
	iii)	python3 a3b.py LEARNING_RATE
	iv)	python3 a4.py LEARNING_RATE
	v)	python3 a5_30.py LEARNING_RATE
	vi)	python3 a5_1000.py LEARNING_RATE
	vii)	python3 a6.py LEARNING_RATE BUFFER_SIZE
	viii)	python3 a7.py LEARNING_RATE BUFFER_SIZE
	ix)	python3 a8.py LEARNING_RATE
LEARNING_RATE and BUFFER_SIZE are numerical hyperparameters. All these scripts save their best model in the model folder, using the script name, hyperparameters, and time-stamp as the model name.

b) Testing:
These commands will test all the models trained for a particular question. During evaluation, you can examine the performance visually by typing 'r' followed by ENTER. Enter any other character to stop rendering.
	i)	python3 test_a3a.py
	ii)	python3 test_a3b.py
	iii)	python3 test_a4.py
	iv)	python3 test_a5_30.py
	v)	python3 test_a5_1000.py
	vi)	python3 test_a6.py
	vii)	python3 test_a7.py
	viii)	python3 test_a8.py


II. PROBLEM B:

While running these files, you can render the environment by typing 'r' followed by ENTER. Enter any other character to stop rendering.

a) Training:
	i)	python3 b1.py GAME_INDEX NUM_EPISODES
	ii)	python3 b2.py GAME_INDEX NUM_EPISODES
	iii)	python3 b34.py GAME_INDEX LEARNING_RATE DISCOUNT_STARTING_FACTOR

GAME_INDEX can be 0, 1 or 2, corresponding to Pong, Ms Pacman, Boxing.
NUM_EPISODES is number of episodes to run for.

LEARNING_RATE and DISCOUNT_STARTING_FACTOR are hyperparameters. DISCOUNT_STARTING_FACTOR controls how quickly epsilon decays from 1 to 0.1. If this parameter is set to 0, epsilon takes ~500,000 steps to reach 0.1; if this parameter is set to 500000, epsilon starts at 0.1, and does not change throughout training.

b34.py saves its best model in the model folder, using the script name, hyperparameters, and time-stamp as the model name.

b) Testing:
	i) python3 test_b4.py GAME_INDEX
Tests each model for a given game for 100 episodes.


DISCLAIMER:
Please note that these scripts are fragile and may not present meaningful or helpful output if supplied with wrong input. Please run them as specifed above, directly from the code folder.
