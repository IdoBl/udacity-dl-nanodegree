import numpy as np

def softmax(L):
    """
    A function that takes as input a list of numbers, and returns
    the list of values given by the softmax function.
    :param L: The input scores
    :return: List of the softmax results for the scores
    """
    softmax_results = []
    softmax_denominator = 0.0
    for score in L:
        softmax_denominator += np.exp(score)
    for score in L:
        softmax_results.append(np.exp(score) / softmax_denominator)
    return softmax_results

