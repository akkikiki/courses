import argparse
from collections import Counter, defaultdict

import random
import numpy
from numpy import median
from sklearn.neighbors import BallTree

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


class Knearest:
    """
    kNN classifier
    """

    def __init__(self, x, y, k=5):
        """
        Creates a kNN instance

        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """

        # You can modify the constructor, but you shouldn't need to.
        # Do not use another data structure from anywhere else to
        # complete the assignment.

        self._kdtree = BallTree(x)
        self._y = y
        self._k = k
        self.median_triggered = defaultdict(int) # For analysis


    def majority(self, item_indices):
        """
        Given the indices of training examples, return the majority label.  If
        there's a tie, return the median of the majority labels (as implemented 
        in numpy).

        :param item_indices: The indices of the k nearest neighbors
        """
        assert len(item_indices) == self._k, "Did not get k inputs"
        # Finish this function to return the most common y label for
        # the given indices.  The current return value is a placeholder 
        # and definitely needs to be changed. 
        #
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.median.html

        nearest_labels = [self._y[x] for x in item_indices]

        labels_dic = defaultdict(int)
        for i in nearest_labels:
            labels_dic[i] += 1

        # You should not assume that the target classes are only two
        labels_occur_sorted = sorted(labels_dic.items(), key= lambda x:x[1], reverse=True)
        top_occuring_label, top_occuring_label_val = labels_occur_sorted[0]

        # Finding if there is a tie
        if len(labels_dic) > 1:
            second_occuring_label, second_occuring_label_val = labels_occur_sorted[1]

            if top_occuring_label_val == second_occuring_label_val:
                self.median_triggered[(top_occuring_label, second_occuring_label)] += 1
                return numpy.median(nearest_labels)

        return top_occuring_label

    def classify(self, example):
        """
        Given an example, classify the example.

        :param example: A representation of an example in the same
        format as training data
        """

        # Finish this function to find the k closest points, query the
        # majority function, and return the predicted label.
        # Again, the current return value is a placeholder 
        # and definitely needs to be changed. 
        dist, indice = self._kdtree.query(example, k=self._k)

        return self.majority(indice[0])

    def confusion_matrix(self, test_x, test_y):
        """
        Given a matrix of test examples and labels, compute the confusion
        matrix for the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.

        :param test_x: Test data representation
        :param test_y: Test data answers
        """

        # Finish this function to build a dictionary with the
        # mislabeled examples.  You'll need to call the classify
        # function for each example.

        # Can handle cases when the returned class is in decimals (e.g. 1.5)
        d = defaultdict(lambda: defaultdict(int))
        data_index = 0

        # To guarantee that the confusion matrix is in 10 rows and 10 columns
        for ii in set(test_y):
            for jj in set(test_y):
                d[ii][jj] = 0

        for xx, yy in zip(test_x, test_y):

            d[yy][self.classify(xx)] += 1

            data_index += 1
            if data_index % 100 == 0:
                print("%i/%i for confusion matrix" % (data_index, len(test_x)))

        return d

    @staticmethod
    def accuracy(confusion_matrix):
        """
        Given a confusion matrix, compute the accuracy of the underlying classifier.
        """

        # You do not need to modify this function

        total = 0
        correct = 0
        for ii in confusion_matrix:
            total += sum(confusion_matrix[ii].values())
            correct += confusion_matrix[ii].get(ii, 0)

        if total:
            return float(correct) / float(total)
        else:
            return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")

    # You should not have to modify any of this code

    if args.limit > 0:
        print("Data limit: %i" % args.limit)
        knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit],
                       args.k)
    else:
        knn = Knearest(data.train_x, data.train_y, args.k)
    print("Done loading data")

    output_conf_matrices = open("confusion_matrices.txt", "a")
    output_conf_matrices.write("k = %d, limit = %d" % (args.k, args.limit) + "\n")
    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    print("\t" + "\t".join(str(x) for x in xrange(10)))
    output_conf_matrices.write("\t" + "\t".join(str(x) for x in xrange(10)) + "\n")
    print("".join(["-"] * 90))
    output_conf_matrices.write("".join(["-"] * 90) + "\n")
    for ii in xrange(10):
        # Outputting confusion matrices for future analysis
        print("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0))
                                       for x in xrange(10)))
        output_conf_matrices.write("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0))
                                       for x in xrange(10)))
        output_conf_matrices.write("\n")
    print("Accuracy: %f" % knn.accuracy(confusion))

    # Outputting accuracies to plot it later.
    output_file = open("accuracies.txt", "a")
    output_file.write(str(args.limit) + "\t" + str(knn.accuracy(confusion)) + "\n")

    # Outputting the occurrences of the median tie-breaker.
    for k, v in sorted(knn.median_triggered.items(), key=lambda x:x[1], reverse=True):
        print("median tie-breaker triggered %d times between %d and %d" % (v, k[0], k[1]))

