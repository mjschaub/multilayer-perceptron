import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier 

training_data = pickle.load(open("train.p","rb"))
test_data = pickle.load(open("test.p","rb"))


for i in range(0, 10):
    training_text = []
    training_label = []
    validation_text = []
    validation_label = []

    classifier = MLPClassifier(activation = 'relu',  max_iter = 500, hidden_layer_sizes = (200,200,200),  alpha = .01,momentum = .9)

    for j in range(len(training_data)):
        if j % 5 ==i:
            validation_text.append(training_data[j][0])
            validation_label.append(training_data[j][1])
        else:
            training_text.append(training_data[j][0])
            training_label.append(training_data[j][1])

    train_text = np.array(training_text)
    train_label = np.array(training_label)
    val_text = np.array(validation_text)
    val_label = np.array(validation_label)




    classifier.fit(train_text, train_label)




    val_num = 0.0
    length = len(val_text)
    for x in range(length):
        if classifier.predict(val_text[x].reshape(1, -1)) == val_label[x]:
            val_num = val_num + 1
    if(length!=0):

    	print ("Val Score ", val_num/(length))




    test_num= 0.0
    length = len(test_data)
    for instance in test_data:
        if  classifier.predict(np.array(instance[0]).reshape(1, -1))[0] == instance[1]:
            test_num = test_num + 1
    if(length!=0):
    	print("test score:" , test_num / (length))
