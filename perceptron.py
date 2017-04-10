import pickle
import numpy as np

if __name__ =='__main__':
	
	training_data = pickle.load(open("train.p", "rb"))
	testing_data = pickle.load(open("test.p", "rb"))

	print(training_data[0])
	
	alpha = 0.01 #learning rate
	n_epoch = 500  # number of epochs
	training_order = 0  #0 for random, 1 for fixed
	weights = 0 #0 for zero weights or 1 for random weights

	for i in training_data:
		train_text = i[0]
		train_label = i[1]
		if weights == 0:
			weights_arr = [0]*len(train_text)
		else:
			weights_arr = np.random.uniform(low=-1.0, high=1.0, size=len(train_text))

		
		
