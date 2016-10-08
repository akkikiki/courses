import argparse
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.base import clone 
import matplotlib.pyplot as plt

np.random.seed(1234)

class FoursAndNines:
    """
    Class to store MNIST data
    """

    def __init__(self, location):

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')

        # Split the data set 
        train_set, valid_set, test_set = cPickle.load(f)

        # Extract only 4's and 9's for training set 
        self.x_train = train_set[0][np.where(np.logical_or( train_set[1]==4, train_set[1] == 9))[0],:]
        self.y_train = train_set[1][np.where(np.logical_or( train_set[1]==4, train_set[1] == 9))[0]]
        self.y_train = np.array([1 if y == 9 else -1 for y in self.y_train])
        
        # Shuffle the training data 
        shuff = np.arange(self.x_train.shape[0])
        np.random.shuffle(shuff)
        self.x_train = self.x_train[shuff,:]
        self.y_train = self.y_train[shuff]

        # Extract only 4's and 9's for validation set 
        self.x_valid = valid_set[0][np.where(np.logical_or( valid_set[1]==4, valid_set[1] == 9))[0],:]
        self.y_valid = valid_set[1][np.where(np.logical_or( valid_set[1]==4, valid_set[1] == 9))[0]]
        self.y_valid = np.array([1 if y == 9 else -1 for y in self.y_valid])
        
        # Extract only 4's and 9's for test set 
        self.x_test  = test_set[0][np.where(np.logical_or( test_set[1]==4, test_set[1] == 9))[0],:]
        self.y_test  = test_set[1][np.where(np.logical_or( test_set[1]==4, test_set[1] == 9))[0]]
        self.y_test = np.array([1 if y == 9 else -1 for y in self.y_test])
        
        f.close()

class AdaBoost:
    def __init__(self, n_learners=20, base=DecisionTreeClassifier(max_depth=1)):
        """
        Create a new adaboost classifier.
        
        Args:
            n_learners (int, optional): Number of weak learners in classifier.
            base (BaseEstimator, optional): Your general weak learner 

        Attributes:
            base (estimator): Your general weak learner 
            n_learners (int): Number of weak learners in classifier.
            alpha (ndarray): Coefficients on weak learners. 
            learners (list): List of weak learner instances. 
        """
        
        self.n_learners = n_learners 
        self.base = base
        self.alpha = np.zeros(self.n_learners)
        self.learners = []
        
    def fit(self, X_train, y_train):
        """
        Train AdaBoost classifier on data. Sets alphas and learners. 
        
        Args:
            X_train (ndarray): [n_samples x n_features] ndarray of training data   
            y_train (ndarray): [n_samples] ndarray of data 
        """

        # TODO 

        # Hint: You can create and train a new instantiation 
        # of your sklearn weak learner as follows 

        # http://grandmaster.colorado.edu/~cketelsen/files/csci5622/videos/lesson09/lesson09.pdf
        w = np.array([1/float(len(y_train))] * len(y_train))
 
        for k in range(self.n_learners):
            h = clone(self.base) # weak learner, decision tree classifier
            h.fit(X_train, y_train, sample_weight=w)
            self.learners.append(h)
            predictions = h.predict(X_train)
            corrects = 0
            
            for i in range(len(y_train)):
                corrects += w[i] * (y_train[i] != predictions[i])

            err = corrects/float(sum(w))
            alpha = 0.5 * np.log((1 - err) / err)
            self.alpha[k] = alpha

            # Updating the weights
            w_new = np.ones(len(y_train))
            for l in range(len(y_train)):
                w_new[l] = w[l] * np.exp(-1 * alpha * y_train[l] * predictions[l])
            sum_w_new = sum(w_new)
            for l in range(len(y_train)):
                w[l] = w_new[l]/sum_w_new
            
    def predict(self, X):
        """
        Adaboost prediction for new data X.
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            
        Returns: 
            [n_samples] ndarray of predicted labels {-1,1}
        """

        # TODO 
        predictions = np.zeros(len(X))
        for i, learner in enumerate(self.learners):
            predictions += self.alpha[i] * learner.predict(X)

        return np.sign(predictions)
    
    def score(self, X, y):
        """
        Computes prediction accuracy of classifier.  
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            y (ndarray): [n_samples] ndarray of true labels  
            
        Returns: 
            Prediction accuracy (between 0.0 and 1.0).
        """

        # TODO 
        y_predicted = self.predict(X)
        correct = [y[i] == y_predicted[i] for i in range(len(y))]

        return sum(correct)/float(len(correct))
    
    def staged_score(self, X, y):
        """
        Computes the ensemble score after each iteration of boosting 
        for monitoring purposes, such as to determine the score on a 
        test set after each boost.
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            y (ndarray): [n_samples] ndarray of true labels  
            
        Returns: 
            [n_learners] ndarray of scores 
        """

        # TODO 
        score = np.zeros(self.n_learners)
        predictions = np.zeros(len(X))

        for k in range(self.n_learners):
            #predictions = np.zeros(len(X))
            #for i in range(k+1):
            predictions += self.alpha[k] * self.learners[k].predict(X)
 
            y_predicted = np.sign(predictions)
            correct = [y[i] == y_predicted[i] for i in range(len(y))]
            score[k] = sum(correct)/float(len(correct))

        return score


def mnist_digit_show(flatimage, outname=None):

	import matplotlib.pyplot as plt

	image = np.reshape(flatimage, (-1,28))

	plt.matshow(image, cmap=plt.cm.binary)
	plt.xticks([])
	plt.yticks([])
	if outname: 
	    plt.savefig(outname)
	else:
	    plt.show()

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='AdaBoost classifier options')
	parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
	parser.add_argument('--n_learners', type=int, default=50,
                        help="Number of weak learners to use in boosting")
	parser.add_argument('--max_depth', type=int, default=1,
                        help="The maximum depth of a tree to use in boosting")
	parser.add_argument('--base_learner', type=str, default="DecisionTreeClassifier",
                        help="The maximum depth of a tree to use in boosting")
	args = parser.parse_args()

	data = FoursAndNines("../data/mnist.pkl.gz")

    # An example of how your classifier might be called
	#clf = AdaBoost(n_learners=50, base=DecisionTreeClassifier(max_depth=1, criterion="entropy"))
        if args.base_learner == "Perceptron":
	    clf = AdaBoost(args.n_learners, base=Perceptron())
        else:
	    clf = AdaBoost(args.n_learners, base=DecisionTreeClassifier(max_depth=args.max_depth, criterion="entropy"))
        clf.fit(data.x_train, data.y_train)
        staged_score_training = clf.staged_score(data.x_train, data.y_train)
        staged_score_testing = clf.staged_score(data.x_test, data.y_test)
        f_out_train = open("%s_learners_%s_max_depth_%s_training" % (args.base_learner, str(args.n_learners), str(args.max_depth)), "w")
        f_out_test = open("%s_learners_%s_max_depth_%s_testing" % (args.base_learner, str(args.n_learners), str(args.max_depth)), "w")
        print(staged_score_training)
        for accuracy in staged_score_training:
            f_out_train.write(str(accuracy) + "\n")
        for accuracy in staged_score_testing:
            f_out_test.write(str(accuracy) + "\n")
