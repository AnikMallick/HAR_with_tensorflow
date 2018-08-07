'''
This is used after training, outputs confusion_matrix
and  'precision','recall','f1 score'
'''
from include import Data_set
from include import Neural_network
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
#Hiding tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
''' 

def plot_confusion(confm,classes,title):
	plt.imshow(confm,cmap = 'gray')

	plt.colorbar()
	thr = confm.max()/2
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks,classes)
	plt.yticks(tick_marks,classes)
	plt.title(title)
	plt.xlabel('Predicted_Labels')
	plt.ylabel('True_Labels')
	for i in range(confm.shape[0]):
		for j in range(confm.shape[1]):
			plt.text(j,i,confm[i,j],horizontalalignment="center",color="white" if confm[i, j] < thr else "black")
	plt.show()

test_data_path = '/UCI HAR Dataset/test'

test = Data_set(path = test_data_path)
test.load_data(features = 'X_test.txt' ,labels = 'y_test.txt')

features, _, labels = test.get_data()
nn = Neural_network()
df = nn.predict(features = features, true_labels= labels)
cm = confusion_matrix(y_true = df['True_Labels'], y_pred = df['Predicted_Labels'])

plot_confusion(cm, [1,2,3,4,5,6], 'confusion_matrix')

true_positive = 0
false_positive = 0
false_negative = 0
for cls in range(6):
	true_positive += cm[cls,cls]
	true_positive /= 6

	false_positive += (sum(cm[:,cls]) - cm[cls,cls])
	false_positive /= 6

	false_negative += (sum(cm[cls,:],1) - cm[cls,cls])
	false_negative /= 6

precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1 = 2 * (precision * recall) / (precision + recall)

print('precision:',precision,', recall:',recall,', f1:',f1)