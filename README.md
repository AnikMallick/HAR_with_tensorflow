# Human Activity Recognition with Tensorflow (Neural Network)

Implementing Neural Network with tensorflow to Recognize Human activity. Here H.A.R. dataset from UCI Machine Learning Repository was used, which has 10299 no of instances and 561 attributes. This database was built from the recordings of 30 subjects performing activities of daily living (ADL) while carrying a waist-mounted smart phone with embedded inertial sensors. In total of six activity (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) was recorded in the dataset. 
A simple neural network of 1 hidden layer with 250 neurons was used to recognize activities with 95.27% accuracy.

## Results:

Obtained an accuracy of 95.27% with the below hyperparameters - 

'No of hidden layers: 1'

No of Nodes: 250

Initial learning rate: 0.001

Hold probability: 0.5

Weight initializer: xavier_initializer

Bias: 0.1

Validation threshold: 0.992


## Observed output -

Test accuracy: 95.27%

Precision: 0.98554

Recall: 0.99085

F1 score: 0.98819

![confm](https://user-images.githubusercontent.com/22342888/43793938-8cd53bb6-9a9a-11e8-8810-30d8c9620532.png)
Figure 1(a)
 
![accuracy_plot](https://user-images.githubusercontent.com/22342888/43794290-6f7b1d82-9a9b-11e8-956d-9d1009f5306c.PNG)
Figure 1(b)
 
Figure 1(c)
 
![acc_table](https://user-images.githubusercontent.com/22342888/43793671-e582a448-9a99-11e8-89b1-b44ddf628181.jpeg)

Figure 1(d)


Figure 1(a) - Confusion Matrix, Figure 1(b) - Accuracy plot on the training data, 
Figure 1(c) - Loss plot on the training data, Figure 1(d) - Table of accuracy of each class on test data




# Reference - 
Data Resource- 
UCI Machine Learning Repository(https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones#)

