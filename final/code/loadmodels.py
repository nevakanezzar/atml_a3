import os

MODEL_FOLDER = '../model/'

def LoadModels(stub):
	
	dir_list = os.listdir(MODEL_FOLDER)
	model_list = []
	print("\nModels:")
	for f in dir_list:
		if stub in f:
			f2 = f.split('.')
			if f2[-1]=='model':
				print(f)
				model_list.append(f)

	return model_list


