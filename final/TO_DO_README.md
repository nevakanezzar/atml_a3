README.md 

Name: Sudhanshu Kasewa
Student Number: 15115014
Course: COMPGI13 - Advanced Topics in Machine Learning
Coursework: Assignment 2

This readme file details the structure of the Assignment 2 submission as well as the commands for running the code.


1. Structure:

The submission is structured as follows:

a) 15115014_assignment2.zip -- Parent zip file, containing the following:
b) code -- folder containing code
c) models -- folder containing trained models. Folder contains subfolders for each model
d) 15115014_report.pdf -- a pdf report of the experiments carried out for the assignment
e) README.md -- this readme file.

The submission expects:
i) the mnist data in a folder called data in the root folder; however, in its absence, it will download a copy at this location. 
ii) the inpainting data files in the data folder. In the absense of these files, task 3 will not run.


2. Python and Tensorflow versions

This assignment was completed using python 3.5, and tensorflow version r0.12.


3. Running the code

The code must be run from the code folder. The code can be run with the following commands:

I. TASK 1 - Classification

a) Single layer LSTM of size 32
	i) Training: python3 task1a_simplelstm.py -t 32
	ii) Testing: python3 task1a_simplelstm.py -s 32

b) Single layer GRU of size 32
	i) Training: python3 task1b_simplegru.py -t 32
	ii) Testing: python3 task1b_simplegru.py -s 32

In the above commands, 32 can be replaced by 64 or 128 for RNNs of those sizes.

c) Three-layered LSTMs of size 32
	i) Training: python3 task1c_layeredlstm.py -t
	ii) Testing: python3 task1c_layeredlstm.py -s

c) Three-layered GRUs of size 32
	i) Training: python3 task1d_layeredgru.py -t
	ii) Testing: python3 task1d_layeredgru.py -s


II TASK 2 - Pixel prediction

a) Single layer GRU of size 32
	i) Training: python3 task2a_simple.py -t 32 1 0.0001
	ii) Testing: python3 task2a_simple.py -s 32 1 0.0001
	iii) In-painting: python3 task2a_simple.py -i 32 1 0.0001

	All three parameters need to be passed for the code to run.

	In the above commands, 32 can be replaced with 64 and 128 for RNNs of those sizes.

	For training the parameter 0.0001 may be replaced by any other learning rate between 0 and 1. This parameter does not affect testing or in-painting, yet must be specified.

	Testing returns only the loss as well as pixel-prediction accuracy of the trained model.

	In-painting samples 100 images at random from the test set, performs 1,10,28 and 300 pixel completions, and generates graphs in the respective model's folder

b) Three-layered GRUs of size 32
	i) Training: python3 task2b_layered.py -t 32 3 0.0001
	ii) Testing: python3 task2b_layered.py -s 32 3 0.0001
	iii) In-painting: python3 task2b_layered.py -i 32 3 0.0001

	The same interface as above applies.


III TASK 3 - In-painting missing pixels

a) One-pixel in-painting:
	i) Single layer GRU of size 32: python3 task3a_one 32 1
		32 above can be replaced with 64 or 128
	ii) 3-layered GRUs of size 32: python task3b_layered_one 32 3

b) 2X2 pixel in-paiting:
	i) Single layer GRU of size 32: python3 task3c_2x2 32 1
		32 above can be replaced with 64 or 128
	ii) 3-layered GRUs of size 32: python task3d_layered_2x2 32 3

	Output of these scripts: 20 completed images are sampled and saved as a png file within the model's folder, along with the dataset augmented with completed in-paintings for all samples.


DISCLAIMER:
Please note that these scripts are fragile and may not present meaningful or helpful output if supplied with wrong input. Please run them as specifed above, directly from the code folder.
