import numpy as np 

kINSP = np.array([(1, 8, +1),
               (7, 2, -1),
               (6, -1, -1),
               (-5, 0, +1),
               (-5, 1, -1),
               (-5, 2, +1),
               (6, 3, +1),
               (6, 1, -1),
               (5, 2, -1)])

kSEP = np.array([(-2, 2, +1),    # 0 - A
              (0, 4, +1),     # 1 - B
              (2, 1, +1),     # 2 - C
              (-2, -3, -1),   # 3 - D
              (0, -1, -1),    # 4 - E
              (2, -3, -1),    # 5 - F
              ])


def weight_vector(x, y, alpha):
    """
    Given a vector of alphas, compute the primal weight vector w. 
    """

    w = np.zeros(len(x[0]))
    # TODO: IMPLEMENT THIS FUNCTION
    # Slide 45 of http://grandmaster.colorado.edu/~cketelsen/files/csci5622/videos/lesson05/lesson05.pdf
    # Computed using the derivation of w
    # for j in range(len(w)):
    #     for i in range(len(alpha)):
    #         w[j] += alpha[i] * y[i] * x[i][j]
    # for j in range(len(w)):
    for i in range(len(alpha)):
        w += alpha[i] * y[i] * x[i]
    return w



def find_support(x, y, w, b, tolerance=0.001):
    """
    Given a set of training examples and primal weights, return the indices 
    of all of the support vectors
    """

    support = set()
    # TODO: IMPLEMENT THIS FUNCTION
    # What does the tolerence mean?
    # Return the indices of all support vector
    for i in range(len(x)):
        distance = y[i] * (np.dot(w, x[i]) + b)
        if np.isclose(distance, np.float64(1.0), atol=tolerance):
            support.add(i) # adding the training example indices that are support vectors.

    return support



def find_slack(x, y, w, b):
    """
    Given a set of training examples and primal weights, return the indices 
    of all examples with nonzero slack as a set.  
    """

    slack = set()
    # TODO: IMPLEMENT THIS FUNCTION
    # How do I compute the slack variable
    # Slide 21 of
    # http://grandmaster.colorado.edu/~cketelsen/files/csci5622/videos/lesson06/lesson06.pdf

    # y[i] * (np.dot(w, x[i]) + b) >= 1 - slack
    # So the examples with y[i] * (np.dot(w, x[i]) + b) < 1
    for i in range(len(x)):
        if y[i] * (np.dot(w, x[i]) + b) < 1:
            slack.add(i)
            # print(i, x)

    return slack


