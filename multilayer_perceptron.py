import pickle
import numpy as np
import sklearn

if __name__ =='__main__':
	
	training_data = pickle.load(open("train.p", "rb"))
	testing_data = pickle.load(open("test.p", "rb"))

	print("training data: ", training_data[0], " length of it: ",len(training_data[0][0])/23)
	
	alpha = 0.01 #learning rate
	n_epoch = 500  # number of epochs
	training_order = 0  #0 for random, 1 for fixed
	weights = 0 #0 for zero weights or 1 for random weights

	perceptrons = dict() #the different perceptron for each label number
	for x in range(n_epoch):
		for i in training_data:
			train_text = i[0]
			train_label = i[1]
			if weights == 0 and x == 0:
				weights_arr = [0]*len(train_text)
				perceptrons[train_label] = weights_arr
			elif x == 0:
				weights_arr = np.random.uniform(low=-1.0, high=1.0, size=len(train_text))
				perceptrons[train_label] = weights_arr
		
			
			# each iteration, for each x we get wrong, update w by adding alpha*error*x to it
			# err is label(x) - guess(x)
			# num = 1 if summation x*w > 0 or 0 if summation x*w is anything else
			# 
			#
			# activation:     if num > 0: guess(x) = 1
    			#		  else: guess(x) = 0
