# ml1

This network is designed to predict handrawn numbers from the MNIST handwritten digits database. Specifically, it uses the csv version found on Kaggle at: 
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv. The current version in this repository training on the given number of iterations and learning rate achieves
an accuracy of approx 95% when trained on 10000 examples. 

The model uses the methods of forward and backward propogation to train the weights and biases on 2 layers (1 hidden) to achieve this result. It also uses the ReLu and 
softmax activation functions. It does not use any machine learning libraries, only pandas and numpy as I felt this would enhance my understanding of machine learning by
approaching it from a low level with few abstractions.
