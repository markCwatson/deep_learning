from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        for j in range(num_classes):
            if j == y[i]:
                continue

            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                
                # Gradient update for incorrect class
                # Need to decrease score for incorrect class
                # Increasing the gradient will achieve this (dL/dW = X)
                # Results in W_j and X[i] being less aligned (for j, i.e. incorrect classes)
                dW[:, j] += X[i]

                # Gradient update for correct class
                # Need to increase score for correct class
                # Decreasing the gradient will achieve this
                dW[:, y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Average over number of training examples
    dW /= num_train

    # Add regularization to the gradient
    # L2 regularization for L includes reg * W^2 term
    # Taking dL/dW of this yields reg * W
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    # Compute scores
    scores = np.dot(X, W)  # Shape: [num_train, num_classes]

    # Compute the correct class scores (shape [num_train, 1])
    correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1)

    # Compute margins
    margins = np.maximum(0, scores - correct_class_scores + 1)  # delta = 1
    margins[np.arange(num_train), y] = 0  # Do not consider correct class in loss

    # Compute loss
    loss = np.sum(margins) / num_train
    loss += reg * np.sum(W * W)  # Regularization term

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Compute gradient
    X_mask = np.zeros(margins.shape)
    X_mask[margins > 0] = 1
    incorrect_counts = np.sum(X_mask, axis=1)  # Count number of incorrect classes
    X_mask[np.arange(num_train), y] -= incorrect_counts
    dW = np.dot(X.T, X_mask) / num_train
    dW += 2 * reg * W  # Regularization term

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
