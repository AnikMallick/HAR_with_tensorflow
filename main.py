'''
Note for use:
	Download the data from 
	https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones#
	And assign train directory to 'train_data_path'
	and assign test directory to 'test_data_path'
'''

from include import Data_set
from include import Neural_network

'''
#Hiding tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
''' 
train_data_path = '/UCI HAR Dataset/train'
test_data_path = '/UCI HAR Dataset/test'

train = Data_set(path = train_data_path)
train.load_data(features = 'X_train.txt' ,labels = 'y_train.txt')
validation_features, validation_onehot , _ = train.get_segment_data(segment_size = 0.20)

test = Data_set(path = test_data_path)
test.load_data(features = 'X_test.txt' ,labels = 'y_test.txt')
holdout_features, _ , holdout_labels = test.get_segment_data(segment_size = 0.02)

nn = Neural_network(n_examples = train.n_examples, 
					n_features = train.n_features, 
					n_classes = train.cls_len)
nn.create_graph(node_list = [250])

nn.train_network_with_while(batch_size = 128, 
							next_batch_func = train.next_batch, 
							validation_f = validation_features, 
							validation_onehot = validation_onehot,
							valiation_thr = 0.992)

#nn.train_network(n_iteration = 200, batch_size = 128, next_batch_func = train.next_batch)

accuracy = nn.test_network(n_examples = test.n_examples, batch_size = 128, next_batch_func = test.next_batch)

df = nn.predict(features = holdout_features, true_labels = holdout_labels)

print(df)
