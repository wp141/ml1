import numpy as np

def ReLU(Z):
    # Iterates through every element in Z
    # If the element > 0, returns element, else returns 0
    return np.maximum(0, Z)

def derivative_ReLU(z):
    return z > 0

def softmax(Z):
    # Creates probability matrix
    softmax = np.exp(Z) / sum(np.exp(Z))
    return softmax

def encode_Y(Y):
    # Y.size refers to the amount of examples, 10 is all the possible answers 0-9
    # np.zeros creates a matrix of these dimensions with 0 in every entry
    encoded = np.zeros((Y.size, 10))

    # Find the entry for the current example (in Y), then the index of the current answer Y and set it to 1
    encoded[np.arange(Y.size), Y] = 1

    # Each row is an example so we transpose it to make each column an example, creating the probability vectors
    encoded = encoded.T

    return encoded
    
