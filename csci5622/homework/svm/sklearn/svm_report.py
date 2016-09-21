from sklearn import svm

# Import the MNIST dataset
# http://scikit-learn.org/stable/modules/svm.html

"""
1.  Use the Sklearn implementation of support vector machines to train a classifier to distinguish 3's from 8's (using the MNIST data from the KNN homework).
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


X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)
print("Prediction Result: %s" % clf.predict([[2., 2.]]))

# get support vectors
print(clf.support_vectors_)
