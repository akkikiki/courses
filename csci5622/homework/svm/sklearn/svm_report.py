import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split

# Import the MNIST dataset
# http://scikit-learn.org/stable/modules/svm.html

"""
1.  Use the Sklearn implementation of support vector machines to train a classifier to distinguish 3's from 8's (using the MNIST data from the KNN homework).
1.  Try at least five values of the regularization parameter _C_ and at least two kernels.  Comment on performance for the varying parameters you chose by either testing on a hold-out set or performing cross-validation. 
1.  Give examples of support vectors from each class when using a linear kernel.
"""

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()

data = Numbers("../data/mnist.pkl.gz")
X = []
y = []
hold_out_data = 100


# Filtering out labels other than 3 or 8
for i in range(len(data.train_y)):
    if data.train_y[i] == 3:
        y.append(-1)
        X.append(data.train_x[i])
    elif data.train_y[i] == 8:
        y.append(1)
        X.append(data.train_x[i])
 

C = [0.01, 0.1, 1.0, 10.0, 100.0]
random_state = np.random.RandomState(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=random_state)

#X = [[0, 0], [1, 1]]
#y = [0, 1]
clf = svm.SVC()
#clf.fit(X[:-hold_out_data], y[:-hold_out_data])
clf.fit(X_train, y_test)

print("Prediction Result: %s" % clf.predict([X[hold_out_data]]))

# get support vectors
print(clf.support_vectors_)
