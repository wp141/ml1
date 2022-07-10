import pandas as pd
import numpy as np
import helper

csv = pd.read_csv('mnist_train.csv')
data = np.array(csv)
m, n = data.shape
np.random.shuffle(data)

#Transposes matrix, now first element is the column vector of the labels
data_train = data[1000:m].T

Y_train = data_train[0] # Labels
X_train = data_train[1:n] # Images
X_train = X_train / 255.

# Selects random weights and biases for first and second layer
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2 

def forward_prop(W1, b1, W2, b2, X):
    # First layer result - dot product of input nodes with the weight, plus the bias
    Z1 = W1.dot(X) + b1

    # Applying activation function ReLU to reduce linearity / add complexity
    A1 = helper.ReLU(Z1) 

    # Output layer result - dot product of A1 result with the weight, plus the bias
    Z2 = W2.dot(A1) + b2 

    # Computes probability column vector using softmax
    A2 = helper.softmax(Z2)

    return Z1, A1, Z2, A2

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    # Encode answer into a probability matrix ( zeroes in all entries except correct entry )
    encoded_Y = helper.encode_Y(Y)

    # Compute errors in 2nd layer and how much the weights and biases contributed to the error
    dZ2 = A2 - encoded_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)

    # Compute errors in 1st layer and how much the weights and biases contributed to the error
    # Multiplying dot product result by derivative of activiation function to undo the original function
    dZ1 = W2.T.dot(dZ2) * helper.derivative_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)

    return dW1, db1, dW2, db2

# Updates params by subtracting the error * learning rate from the previous weight / bias
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1

    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2

def get_predictions(A2):
    # Gets the current prediction based on the network probability output A2
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0 and i != (iterations - 1):
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
            
    print(f"Training finished with an accuracy of: {get_accuracy(predictions, Y)}")

    # Returns trained network's weights and biases
    return W1, b1, W2, b2

if __name__ == "__main__":
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.25, 1000)
