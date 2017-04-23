import pickle
import numpy as np

if __name__ =='__main__':
	
	training_data = pickle.load(open("train.p", "rb"))
	testing_data = pickle.load(open("test.p", "rb"))

	print("training data: ", training_data[0], " length of it: ",len(training_data))
	alpha_constant = True
	alpha = 0.01 #learning rate
	n_epoch = 500  # number of epochs
	training_order = 0  #0 for random, 1 for fixed
	weights = 0 #0 for zero weights or 1 for random weights
	
	validation_data = training_data[4000:]
	perceptrons = dict() #the different perceptron for each label number
	finished_training = False
	for x in range(n_epoch):

		for i in range(0,4000):
			train_text = training_data[i][0]
			train_label = training_data[i][1]
			print(train_text)
			print(train_label)
			weights_arr = ()
			if weights == 0 and x == 0 and train_label is not perceptrons.keys():
				weights_arr = ([0]*len(train_text),1)
				perceptrons[train_label] = weights_arr
			elif x == 0 and train_label is not perceptrons.keys():
				weights_arr = (np.random.uniform(low=-1.0, high=1.0, size=len(train_text)),1)
				perceptrons[train_label] = weights_arr
			
			#for j in range(len(weights_arr)):
			curr_guess = np.sign(weights_arr[0].dot(train_text)+weights_arr[1])
			err = train_label - curr_guess
			print(err)
			if err is not 0:
				delta_bias = err
				weights_arr[1] += delta_bias
				weight_arr[0] += alpha*err*train_text
		
		
		num_correct = 0
		total_num_indices = 0
		for i in validation_data:
			trained_data_to_check = perceptrons[i[1]]
			validation_arr = i[0]
			for j in range(len(trained_data_to_check)):
				if trained_data_to_check[j] == validation_arr[j]:
					num_correct+=1
				total_num_indices +=1

		if (num_correct/total_num_indices) >= .8:
			finished_training = True
			print("perceptron finished training, reached 80%")	
			break
		if alpha_constant is False:
			alpha *= 1/x

	'''
	for each epoch
    for each training data instance
        propagate error through the network
        adjust the weights
        calculate the accuracy over training data
    for each validation data instance
        calculate the accuracy over the validation data
    if the threshold validation accuracy is met
        exit training
    else
        continue training
	'''


			# each iteration, for each x we get wrong, update w by adding alpha*error*x to it
			# err is label(x) - guess(x)
			# num = 1 if summation x*w > 0 or 0 if summation x*w is anything else
			# delta_bias = err
			# new_bias = bias+delta_bias
			#
			# activation:     if num > 0: guess(x) = 1
    			#		  else: guess(x) = 0





