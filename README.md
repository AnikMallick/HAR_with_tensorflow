# Human Activity Recognition with Tensorflow (Neural Network)

Implementing Neural Network with tensorflow to Recognize Human activity. Here H.A.R. dataset from UCI Machine Learning Repository was used, which has 10299 no of instances and 561 attributes. This database was built from the recordings of 30 subjects performing activities of daily living (ADL) while carrying a waist-mounted smart phone with embedded inertial sensors. In total of six activity (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) was recorded in the dataset. 
A simple neural network of 1 hidden layer with 250 neurons was used to recognize activities with 95.27% accuracy.

# Results:

Obtained an accuracy of 95.27% with the below hyperparameters - 

'No of hidden layers: 1'

No of Nodes: 250

Initial learning rate: 0.001

Hold probability: 0.5

Weight initializer: xavier_initializer

Bias: 0.1

Validation threshold: 0.992


### Observed output -

Test accuracy: 95.27%

Precision: 0.98554

Recall: 0.99085

F1 score: 0.98819


 
Figure 1(a)
 

Figure 1(b)
 
Figure 1(c)
 
Figure 1(d)


Figure 1(a) - Confusion Matrix, Figure 1(b) - Accuracy plot on the training data, 
Figure 1(c) - Loss plot on the training data, Figure 1(d) - Table of accuracy of each class on test data




# Reference - 
Data Resource- UCI Machine Learning Repository(https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones#)

