import pickle
import numpy as np

if __name__ =='__main__':
	
	training_data = pickle.load(open("train.p", "rb"))
	testing_data = pickle.load(open("test.p", "rb"))

	#print("training data: ", training_data[0], " length of it: ",len(training_data))
	alpha_constant = True
	alpha = 1 #learning rate
	n_epoch = 500  # number of epochs
	weights = 1 #0 for zero weights or 1 for random weights
	
	validation_data = training_data[4000:]
	perceptrons = dict() #the different perceptron for each label number
	finished_training = False
	for i in range(0,10):	
		weights_arr = []
		if weights == 0:
			weights_arr = [np.array([0.0]*len(training_data[0][0])),1]
			perceptrons[i] = weights_arr
		else:
			weights_arr = [np.random.uniform(low=0.0, high=1.0, size=len(training_data[0][0])),1]
			perceptrons[i] = weights_arr
	num_valid = 0
	for x in range(n_epoch):
		for i in range(0,4000):
			train_text = training_data[i][0]
			train_label = int(training_data[i][1])
			
			max_percep = perceptrons[0][0].dot(train_text)+perceptrons[0][1]
			max_percep_idx = 0
			for j in range(0,10):
				temp = perceptrons[j][0].dot(train_text)+perceptrons[j][1]
				if temp > max_percep:
					max_percep_idx = j
					max_percep = temp
				
			percep_x = max_percep
			
			if max_percep_idx != train_label:
				delta_bias = alpha
				perceptrons[train_label][1] += delta_bias
				perceptrons[max_percep_idx][1] -= delta_bias
				for y in range(len(perceptrons[train_label][0])):
					perceptrons[train_label][0][y] += alpha*train_text[y]
					perceptrons[max_percep_idx][0][y] -= alpha*train_text[y]
		
		
		num_correct = 0.0
		total_num_indices = 0.0
		for i in validation_data:
			max_percep = perceptrons[0][0].dot(i[0])+perceptrons[0][1]
			max_percep_idx = 0
			for j in range(0,10):
				temp = perceptrons[j][0].dot(i[0])+perceptrons[j][1]
				if temp > max_percep:
					max_percep_idx = j
					max_percep = temp
				
			percep_x = max_percep
			if max_percep_idx == i[1]:
				num_correct+=1
			total_num_indices+=1

		
		if (num_correct/total_num_indices) >= .8:
			finished_training = True
			num_valid+=1
			print(num_correct/total_num_indices)
			if num_valid > 20:
				print "perceptron finished training, reached 80% on validation data with number of epochs being:",x	
				break
		if alpha_constant is False:
			alpha = 100/(100+x)


	num_correct = 0.0
	total_num_indices = 0.0
	for i in testing_data:
		max_percep = perceptrons[0][0].dot(i[0])+perceptrons[0][1]
		max_percep_idx = 0
		for j in range(0,10):
			temp = perceptrons[j][0].dot(i[0])+perceptrons[j][1]
			if temp > max_percep:
				max_percep_idx = j
				max_percep = temp
	
		percep_x = max_percep
		if max_percep_idx == i[1]:
			num_correct+=1
		total_num_indices+=1

	
	if (num_correct/total_num_indices) >= .8:
		finished_training = True
		print("perceptron passed testing data with:")
		print (num_correct/total_num_indices)*100.0,'%'
	else:
		print("perceptron failed testing data with:")
		print (num_correct/total_num_indices)*100.0,'%'



