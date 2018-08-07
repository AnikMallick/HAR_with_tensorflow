import numpy as np 
import tensorflow as tf
import pandas as pd

class Data_set(object):
	def __init__(self, path):
		'''Dataset Class

		Args:
			path: path of the data, path must contain both 
				features and labels in different file
		'''
		self._path = path
		self._cls = [0]
		self._cls_len = len(self._cls)
		self._n_features = None
		self._n_examples = None
		self.__data = None
		self.__labels = None
		self._index_in_epoch = 0

	@property
	def n_examples(self):
		return self._n_examples

	@property
	def cls(self):
		return self._cls

	@property
	def path(self):
		return self._path

	@property
	def cls_len(self):
		return self._cls_len

	@property
	def n_features(self):
		return self._n_features

	@path.setter
	def path(self, args):
		self._path = args
	

	@staticmethod
	def onehot_encoder(classes,actual_labels):
		'''Helper for encoding the labels to Onehot array

		Args:
			classes: List of classes
			actual_labels: labels that need to be encoded

		Return:
			1. nd array of onehot encoded labels
			2. onehot encoding dict
		'''
		onehot_labels = []
		cls_onehot_dict = {}
		for clss in classes:
			temp_list = [0]*(len(classes))
			temp_list[classes.index(clss)] = 1
			cls_onehot_dict[clss] = temp_list
		
		for labels in actual_labels:
			onehot_labels.append(cls_onehot_dict[labels])
		return np.array(onehot_labels), cls_onehot_dict


	def load_data(self,features,labels):
		'''Loads the data(features and labels)

		Args:
			features: features file name
			labels: labels file name

		Note:
			This method also sets few properties
			Data is shuffled by default
			Uses numpy.genfromtxt to read data files
		'''
		features = self._path + '\\' + features
		labels = self._path + '\\' + labels
		self.__data = np.genfromtxt(features)
		self.__labels = np.genfromtxt(labels)

		#setting properties
		self._n_examples = self.__data.shape[0]
		self._n_features = self.__data.shape[1]
		temp = list(np.unique(self.__labels))
		self._cls.extend(temp)
		self._cls_len = len(self._cls)

		#shuffle
		idx = np.arange(self._n_examples)
		np.random.shuffle(idx)
		self.__data = self.__data[idx]
		self.__labels = self.__labels[idx]

		self.__onehot_labels, self._onehot_encoder_dict = Data_set.onehot_encoder(self._cls, self.__labels)

		print('\nData loading complete from : \n{0}\nand\n{1}\n'.format(features, labels))

	def next_batch(self,batch_size):
		'''This method is passed to the Neural_network class

		Args:
			batch_size: size of the batch 

		Return:
			1. nd array of batch features
			2. nd array of batch onehot labels
			3. nd array of batch labels

		Note:
			This method can be passed to Neural_network, 
			for it to access the data in batches
			At the start of every batch Data is shuffled

		Raises:
			ValueError: If batch size is grater than no of examples
		'''
		if batch_size>self._n_examples:
			raise ValueError('Batch Size cannot be more no of examples')

		#At the start of every batch Data is shuffled
		if self._index_in_epoch == 0:
			idx = np.arange(self._n_examples)
			np.random.shuffle(idx)

			self.__data = self.__data[idx]
			self.__labels = self.__labels[idx]
			self.__onehot_labels = self.__onehot_labels[idx]

		start = self._index_in_epoch
		try:
			assert batch_size< self._n_examples - self._index_in_epoch
			self._index_in_epoch += batch_size
			end = self._index_in_epoch
		except AssertionError:
			end = self._n_examples
			self._index_in_epoch = 0
		
		return self.__data[start:end], self.__onehot_labels[start:end], self.__labels[start:end]

	def get_data(self):
		'''Method for getting all the data

		Return:
			1. nd array of full features
			2. nd array of full onehot labels
			3. nd array of full labels
		'''
		return self.__data, self.__onehot_labels, self.__labels
	
	def get_segment_data(self, segment_size = 0.05, delete_rows = True):
		'''Method to get a segment of data

		Args:
			segment_size: size of the segment needed
			delete_rows: BOOL, if Ture the rows returned in the segment will be deleted

		Return:
			1. nd array of segment features
			2. nd array of segment onehot labels
			3. nd array of segment labels

		Note:
			Returned data is unbiased as it will select equal no of rows for each class
			Returned data is shuffled
		'''
		n_examples_per_class = int(np.floor((self._n_examples * segment_size) / (self._cls_len - 1)))
		segment_n_examples = n_examples_per_class * (self._cls_len - 1)

		feature_to_return = []
		onehot_labels_to_return = []
		labels_to_return = []
		element_index_list = []
		for cls in self._cls:
			element_index = np.nonzero(self.__labels == cls)[0][:n_examples_per_class]
			element_index_list.extend(list(element_index))
			for idx in element_index:
				feature_to_return.append(self.__data[idx])
				onehot_labels_to_return.append(self.__onehot_labels[idx])
				labels_to_return.append(self.__labels[idx])

		#deleting rows from full data if delete_rows == True
		if delete_rows:
			for idx in sorted(element_index_list,reverse = True):
				self.__data = np.delete(self.__data,idx,axis = 0)
				self.__onehot_labels = np.delete(self.__onehot_labels,idx,axis = 0)
				self.__labels = np.delete(self.__labels,idx,axis = 0)
		
			self._n_examples = self.__data.shape[0]

			idx = np.arange(self._n_examples)
			np.random.shuffle(idx)
			self.__data = self.__data[idx]
			self.__labels = self.__labels[idx]
			self.__onehot_labels = self.__onehot_labels[idx]

		feature_to_return = np.array(feature_to_return)
		onehot_labels_to_return = np.array(onehot_labels_to_return)
		labels_to_return = np.array(labels_to_return)

		idx_segment = np.arange(segment_n_examples)
		np.random.shuffle(idx_segment)
		feature_to_return = feature_to_return[idx_segment]
		onehot_labels_to_return = onehot_labels_to_return[idx_segment]
		labels_to_return = labels_to_return[idx_segment]

		return feature_to_return, onehot_labels_to_return, labels_to_return


###############################################################################


class Neural_network(object):
	def __init__(self, n_examples = None, n_features = None, n_classes = None):
		'''Neural_network

		Args:
			n_examples: no of examples in training dataset,
						can be kept as none if restoring from old ckpt
			n_features: no of features in training dataset,
						can be kept as none if restoring from old ckpt
			n_classes: no classes in this experiment

			Note:
				Create directory for Saver and tensorboard
				and assign them to 
				self._summary_dicentory and self._save_dicentory
		'''
		self._n_examples = n_examples
		self._n_features = n_features
		self._n_classes = n_classes
		self._dtype = tf.float32
		self._hold_prob_value = 0.5
		self._print_parameter = 50 #parameter

		self._summary_dicentory = '/log/tensorboard'
		self._save_dicentory = '/log/saver/'

	@property
	def summary_dicentory(self):
		return self._summary_dicentory

	@property
	def save_dicentory(self):
		return self._save_dicentory

	@property
	def print_parameter(self):##
		return self._print_parameter

	@property
	def hold_prob_value(self):
		return self._hold_prob_value	
	
	@summary_dicentory.setter
	def summary_dicentory(self, args):
		self._summary_dicentory = args
	
	@save_dicentory.setter
	def save_dicentory(self, args):
		self._save_dicentory = args
	
	@print_parameter.setter##
	def print_parameter(self, args):
		self._print_parameter = args
	
	@hold_prob_value.setter
	def hold_prob_value(self, args):
		self._hold_prob_value = args


	@staticmethod
	def __variable_on_cpu(name, shape, initializer, dtype):
		'''Helper for creating variable on cpu

		Args:
			name: name of the variable
			shape: shape of the variable
			initializer: initializar to be used to initialize the variable
			dtype: data type of the variable

		Return:
			variable tensor
		'''
		with tf.device('/cpu:0'):
			var = tf.get_variable(name = name, shape = shape, initializer = initializer, dtype = dtype)
		return var
	
	@staticmethod
	def create_layer(inputs, n_inputs, n_nodes, dtype, name = 'fc', use_relu = True):
		'''Helper for creating fully connecting layers

		Args:
			inputs: intups to this layer
			n_inputs: no of input features
			n_nodes: no of nodes in this layer
			dtype: data type
			name:name of this layer
			use_relu: BOOL, for enableling or disbeling relu activation

		Return:
			Layer Tensor
		'''
		with tf.name_scope(name):
			weights = Neural_network.__variable_on_cpu(name = name + '_W',
														dtype = dtype,
														shape = [n_inputs,n_nodes],
														initializer = tf.contrib.layers.xavier_initializer())
			biases = Neural_network.__variable_on_cpu(name = name + '_B',
														shape = [n_nodes],
														initializer=tf.constant_initializer(0.1),
														dtype = dtype)

			layer = tf.matmul(inputs,weights) + biases

			if use_relu:
				layer = tf.nn.relu(layer)

			Neural_network.__add_summary(weights, biases, layer, use_relu)
		return layer

	@staticmethod
	def __add_summary(weights, biases, layer, use_relu):
		'''Helper for adding summary

		Args:
			weights: weights for histogram
			biases: biases for histogram
			layer: layer for histogram of activation and Sparsity
			use_relu: BOOL
		'''
		tf.summary.histogram('Weights',weights)
		tf.summary.histogram('Biases',biases)
		if use_relu:
			tf.summary.histogram('Activation',layer)
			tf.summary.scalar('Sparsity',tf.nn.zero_fraction(layer))

	def create_graph(self, node_list):
		'''Method for creating graph

		Args:
			node_list: list of nodes for hidden layers
		'''
		self._n_layers = len(node_list)
		self._node_list = node_list

		layer_list = []
		layer_name_list = []

		for i in range(self._n_layers):
			layer_name_list.append('layer_' + str(i+1))

		self.__features = tf.placeholder(tf.float32,[None, self._n_features],name = 'features')
		self.__onehot_labels = tf.placeholder(tf.float32, [None, self._n_classes], name = 'onehot_labels')
		self.__hold_prob = tf.placeholder(tf.float32, name = 'hold_prob')

		label_true = tf.argmax(self.__onehot_labels, axis = 1)

		#fully connected layer
		for layer in range(self._n_layers):
			if layer != 0:
				inputs = layer_list[layer - 1]
				n_inputs = self._node_list[layer - 1]
			else:
				inputs = self.__features
				n_inputs = self._n_features

			layer_list.append(Neural_network.create_layer(inputs = inputs, 
															n_inputs = n_inputs, 
															n_nodes = self._node_list[layer], 
															dtype = self._dtype, 
															name = layer_name_list[layer]))
			#dropout
			layer_list[layer] = tf.nn.dropout(layer_list[layer], 
												keep_prob = self.__hold_prob)

		#output layer
		output_layer = Neural_network.create_layer(inputs = layer_list[-1], 
													n_inputs = self._node_list[-1], 
													n_nodes = self._n_classes, 
													dtype = self._dtype, 
													name = 'output_layer',
													 use_relu = False)
		onehot_labels_pred = tf.nn.softmax(output_layer)
		self.__labels_pred = tf.argmax(onehot_labels_pred, axis = 1, name = 'predicted_labels')

		#loss calculation
		with tf.name_scope('loss'):
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = output_layer,
																	labels = self.__onehot_labels)
			self.__loss_op = tf.reduce_mean(cross_entropy, name = 'model_loss')

		#optimization or minimizing loss
		with tf.name_scope('optimizer'):
			self.__train_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(self.__loss_op)

		#calculation of accuracy
		with tf.name_scope('accuracy'):
			correct_pred = tf.equal(label_true, self.__labels_pred)
			self.__accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		self.__accuracy_op = tf.identity(self.__accuracy_op, name="model_accuracy")

		#setting summary for loss and accuracy
		tf.summary.scalar('Loss', self.__loss_op)
		tf.summary.scalar('Accuracy', self.__accuracy_op)

		#creating saver to save trained model
		self.__saver_obj = tf.train.Saver()

	def train_network(self, n_iteration, batch_size, next_batch_func):
		'''Method to train the network with no of iteration

		Args:
			n_iteration: no of iterattion to train
			batch_size: batch size of the data for next_batch_func
			next_batch_func: next_batch function of trainning data
		'''
		acc_list_print = []
		loss_list_print = []

		n_batches = int(np.ceil(self._n_examples / batch_size))
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			marged_summary = tf.summary.merge_all()
			writer = tf.summary.FileWriter(self._summary_dicentory)
			writer.add_graph(sess.graph)

			for epoch in range(n_iteration):
				for batch in range(n_batches):
					batch_X, batch_onehot, _ = next_batch_func(batch_size = batch_size)
					train_feed_dict = {self.__features : batch_X,
										self.__onehot_labels : batch_onehot,
										self.__hold_prob : self._hold_prob_value}

					#training the network
					sess.run(self.__train_op, feed_dict = train_feed_dict)
					if (epoch+1) % 5 == 0:
						s = sess.run(marged_summary, feed_dict = train_feed_dict)
						writer.add_summary(s, epoch)

					if (epoch+1) % self._print_parameter == 0:
						temp_acc, temp_loss = sess.run([self.__accuracy_op, self.__loss_op], feed_dict = train_feed_dict)
						acc_list_print.append(temp_acc)
						loss_list_print.append(temp_loss)

				if (epoch+1) % self._print_parameter == 0:
					accuracy = sum(acc_list_print) / n_batches
					loss = sum(loss_list_print) / n_batches
					msg = 'Epoch: {0}, Completed out of: {1}, Accuracy in this epoch: {2}, Loss in this epoch: {3}'
					print(msg.format(epoch + 1, n_iteration, accuracy, loss))

				acc_list_print = []
				loss_list_print = []

			#saving the model
			self.__saver_obj.save(sess = sess, save_path = self._save_dicentory)

	def __load_graph_from_ckpt(self,sess, load_model_path):
		'''Helper to restore the model/session

		Args:
			sess: session needed to be restored
			load_model_path: path for the ckpt files
		'''
		new_saver = tf.train.import_meta_graph(load_model_path + '.meta')
		new_saver.restore(sess, tf.train.latest_checkpoint(load_model_path))
		graph = tf.get_default_graph()

		self.__features = graph.get_tensor_by_name('features:0')
		self.__onehot_labels = graph.get_tensor_by_name('onehot_labels:0')
		self.__hold_prob = graph.get_tensor_by_name('hold_prob:0')

		self.__labels_pred = graph.get_tensor_by_name('predicted_labels:0')
		self.__accuracy_op = graph.get_tensor_by_name('model_accuracy:0')

	def test_network(self, n_examples, batch_size, next_batch_func, load_model_path = None):
		'''Method for testing the network

		Args:
			n_examples: no of examples in test dataset
			batch_size: batch size for test data
			next_batch_func: next_batch function
			load_model_path: specify path of ckpt, if set as none
							it will load from save_dicentory

		'''
		acc_list_print = []
		n_batches = int(np.ceil(n_examples / batch_size))
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			if load_model_path is None:
				load_model_path = self._save_dicentory

			#load saved model and restore session
			self.__load_graph_from_ckpt(sess,load_model_path)

			for batch in range(n_batches):
				batch_X, batch_onehot, _ = next_batch_func(batch_size = batch_size)
				test_feed_dict = {self.__features : batch_X,
								self.__onehot_labels : batch_onehot,
								self.__hold_prob : 1.0}
				temp_acc= sess.run(self.__accuracy_op, feed_dict = test_feed_dict)
				acc_list_print.append(temp_acc)
			accuracy = sum(acc_list_print) / n_batches

			print('test accuracy: {0}'.format(accuracy))

			return accuracy

	def predict(self, features, true_labels, load_model_path = None):
		'''Method for predicting

		Args:
			features: features for prediction
			true_labels: actual labels of the given features
			load_model_path: specify path of ckpt, if set as none
							it will load from save_dicentory

		Return:
			pandas dataframe with columns: 'True Labels', 'Predicted Labels'
		'''
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			if load_model_path is None:
				load_model_path = self._save_dicentory

			#load saved model and restore session
			self.__load_graph_from_ckpt(sess,load_model_path)

			test_feed_dict = {self.__features : features,
								self.__hold_prob : 1}

			labels_pred = sess.run(self.__labels_pred, feed_dict = test_feed_dict)
			labels_pred = np.array(labels_pred)

		return_data = np.concatenate([true_labels.reshape([-1,1]), labels_pred.reshape([-1,1])],
									axis = 1)
		return_dataframe = pd.DataFrame(data = return_data, columns = ['True_Labels', 'Predicted_Labels'])

		return return_dataframe

	def train_network_with_while(self, batch_size, next_batch_func,validation_f, validation_onehot,valiation_thr = 0.98):
		'''Method to train the network with while loop

		Args:
			batch_size: batch size of the data for next_batch_func
			next_batch_func: next_batch function of training data
			validation_f: validation features
			validation_onehot: validation onehot array
			valiation_thr: Validation accuracy threshold

		Note:
			While loop break condition: validation accuracy threshold or 
										max no of iteration
		'''

		#max no of iteration
		break_epoch = 500

		acc_list_print = []
		loss_list_print = []

		n_batches = int(np.ceil(self._n_examples / batch_size))
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			marged_summary = tf.summary.merge_all()
			writer = tf.summary.FileWriter(self._summary_dicentory)
			writer.add_graph(sess.graph)

			validation_acc = 0
			validation_feed_dict = {self.__features : validation_f,
									self.__onehot_labels : validation_onehot,
									self.__hold_prob : 1}
			epoch = 0

			while True:
				for batch in range(n_batches):
					batch_X, batch_onehot, _ = next_batch_func(batch_size = batch_size)
					train_feed_dict = {self.__features : batch_X,
										self.__onehot_labels : batch_onehot,
										self.__hold_prob : self._hold_prob_value}

					#training
					sess.run(self.__train_op, feed_dict = train_feed_dict)
					if (epoch+1) % 5 == 0:
						s = sess.run(marged_summary, feed_dict = train_feed_dict)
						writer.add_summary(s, epoch)

					if (epoch+1) % self._print_parameter == 0:
						train_check_feed_dict = {self.__features : batch_X,
												self.__onehot_labels : batch_onehot,
												self.__hold_prob : 1.0}
						temp_acc, temp_loss = sess.run([self.__accuracy_op, self.__loss_op], feed_dict = train_check_feed_dict)
						acc_list_print.append(temp_acc)
						loss_list_print.append(temp_loss)

				if (epoch+1) % self._print_parameter == 0:
					accuracy = sum(acc_list_print) / n_batches
					loss = sum(loss_list_print) / n_batches
					msg = 'Epoch: {0}, Accuracy in this epoch: {1}, Loss in this epoch: {2}'
					print(msg.format(epoch + 1, accuracy, loss))
					print('validation acc = ',validation_acc )

				acc_list_print = []
				loss_list_print = []
				epoch += 1
				if epoch % 5 ==0:
					validation_acc = sess.run(self.__accuracy_op, feed_dict = validation_feed_dict)

					#break condition
					if validation_acc > valiation_thr:
						break

				#break condition
				if epoch == break_epoch:
					break

			self.__saver_obj.save(sess = sess, save_path = self._save_dicentory)
