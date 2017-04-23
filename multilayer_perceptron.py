import pickle
import numpy as np
import sklearn
import tensorflow as tf

if __name__ =='__main__':
'''	
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
'''
	#tf.contrib.learn.DNNClassifier
	# Parameters
	learning_rate = 0.001
	training_epochs = 15
	batch_size = 100
	display_step = 1

	# Network Parameters
	n_hidden_1 = 256 # 1st layer number of features
	n_hidden_2 = 256 # 2nd layer number of features
	n_input = 784 # MNIST data input (img shape: 28*28)
	n_classes = 10 # MNIST total classes (0-9 digits)
	 
	# tf Graph input
	x = tf.placeholder("float", [None, n_input])
	y = tf.placeholder("float", [None, n_classes])


	def multilayer_perceptron(x, weights, biases):
		# Hidden layer with ReLU activation
		layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
		layer_1 = tf.nn.relu(layer_1)
		# Hidden layer with ReLU activation
		layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
		layer_2 = tf.nn.relu(layer_2)
		# Output layer with linear activation
		out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
		return out_layer
		 
		# Store layers weight &amp; bias
		weights = {
		'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
		'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
		'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
		}
		biases = {
		'b1': tf.Variable(tf.random_normal([n_hidden_1])),
		'b2': tf.Variable(tf.random_normal([n_hidden_2])),
		'out': tf.Variable(tf.random_normal([n_classes]))
		}
		 
		# Construct model
		pred = multilayer_perceptron(x, weights, biases)

	weights = {
	    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
	}
	biases = {
	    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	    'out': tf.Variable(tf.random_normal([n_classes]))
	}

	# Construct model
	pred = multilayer_perceptron(x, weights, biases)

	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	# Initializing the variables
	init = tf.global_variables_initializer()

	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)

		# Training cycle
		for epoch in range(training_epochs):
			avg_cost = 0.
			total_batch = int(mnist.train.num_examples/batch_size)
			# Loop over all batches
			for i in range(total_batch):
				batch_x, batch_y = mnist.train.next_batch(batch_size)
				# Run optimization op (backprop) and cost op (to get loss value)
				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
									  y: batch_y})
				# Compute average loss
				avg_cost += c / total_batch
			# Display logs per epoch step
			if epoch % display_step == 0:
				print("Epoch:", '%04d' % (epoch+1), "cost=", \
					"{:.9f}".format(avg_cost))
		print("Optimization Finished!")

		# Test model
		correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
		# Calculate accuracy
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
	
