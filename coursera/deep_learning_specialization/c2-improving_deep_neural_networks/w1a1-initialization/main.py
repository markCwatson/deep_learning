#!/usr/bin/env python
# coding: utf-8

# # Initialization

import sys
import numpy as np
import matplotlib.pyplot as plt
from init_utils import compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

"""
Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

Arguments:
X -- input data, of shape (2, number of examples)
Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
learning_rate -- learning rate for gradient descent 
num_iterations -- number of iterations to run gradient descent
print_cost -- if True, print the cost every 1000 iterations
initialization -- flag to choose which initialization to use ("zeros","random" or "he")

Returns:
parameters -- parameters learnt by the model
"""
def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):     
    grads = {}
    costs = [] # to keep track of the loss
    m = X.shape[1] # number of examples
    layers_dims = [X.shape[0], 10, 5, 1]
    
    # Initialize parameters dictionary.
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    # Loop (gradient descent)

    for i in range(num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        a3, cache = forward_propagation(X, parameters)
        
        # Loss
        cost = compute_loss(a3, Y)

        # Backward propagation.
        grads = backward_propagation(X, Y, cache)
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
            
    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

"""
Arguments:
layer_dims -- python array (list) containing the size of each layer.

Returns:
parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                b1 -- bias vector of shape (layers_dims[1], 1)
                ...
                WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                bL -- bias vector of shape (layers_dims[L], 1)
"""
def initialize_parameters_zeros(layers_dims):   
    parameters = {}
    L = len(layers_dims) # number of layers in the network
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters

"""
Arguments:
layer_dims -- python array (list) containing the size of each layer.

Returns:
parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                b1 -- bias vector of shape (layers_dims[1], 1)
                ...
                WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                bL -- bias vector of shape (layers_dims[L], 1)
"""
def initialize_parameters_random(layers_dims):    
    np.random.seed(3)               # This seed makes sure your "random" numbers will be the as ours
    parameters = {}
    L = len(layers_dims)            # integer representing the number of layers
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters

"""
Arguments:
layer_dims -- python array (list) containing the size of each layer.

Returns:
parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                b1 -- bias vector of shape (layers_dims[1], 1)
                ...
                WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                bL -- bias vector of shape (layers_dims[L], 1)
"""
def initialize_parameters_he(layers_dims):    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers
     
    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2./layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        
    return parameters

def main() -> int:
    train_X, train_Y, test_X, test_Y = load_dataset()

    ## First try zero initialization

    # train model on 15,000 iterations using zeros initialization.
    parameters = initialize_parameters_zeros([3, 2, 1])
    parameters = model(train_X, train_Y, initialization = "zeros")
    predictions_train = predict(train_X, train_Y, parameters)
    predictions_test = predict(test_X, test_Y, parameters)

    # The performance is terrible, the cost doesn't decrease, and the algorithm performs no better than random guessing. Why? Take a look at the details of the predictions and the decision boundary:
    print (predictions_train)
    print(predictions_test)
    plt.title("Model with Zeros initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5,1.5])
    axes.set_ylim([-1.5,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

    ## now use random initialization

    # Run the following code to train your model on 15,000 iterations using random initialization.
    parameters = initialize_parameters_random([3, 2, 1])
    parameters = model(train_X, train_Y, initialization = "random")
    predictions_train = predict(train_X, train_Y, parameters)
    predictions_test = predict(test_X, test_Y, parameters)

    # gives noticeably better accuracy
    print (predictions_train)
    print(predictions_test)
    plt.title("Model with large random initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5,1.5])
    axes.set_ylim([-1.5,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

    ## Use He initialization

    # Run the following code to train your model on 15,000 iterations using He initialization.
    parameters = initialize_parameters_he([2, 4, 1])
    parameters = model(train_X, train_Y, initialization = "he")
    predictions_train = predict(train_X, train_Y, parameters)
    predictions_test = predict(test_X, test_Y, parameters)

    # best
    print (predictions_train)
    print(predictions_test)
    plt.title("Model with He initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5,1.5])
    axes.set_ylim([-1.5,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

if __name__ == '__main__':
    sys.exit(main())
