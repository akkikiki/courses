# Author: Yoshinari Fujinuma
import random
import argparse
import math

from numpy import zeros, sign 
from math import exp, log
from collections import defaultdict
import numpy as np


kSEED = 1735
kBIAS = "BIAS_CONSTANT"

random.seed(kSEED)


def sigmoid(score, threshold=20.0):
    """
    Prevent overflow of exp by capping activation at 20.

    :param score: A real valued number to convert into a number between 0 and 1
    """

    if abs(score) > threshold:
        score = threshold * sign(score)

    return 1.0 / (1.0 + exp(-score))


class Example:
    """
    Class to represent a logistic regression example
    """
    def __init__(self, label, words, vocab, df):
        """
        Create a new example

        :param label: The label (0 / 1) of the example
        :param words: The words in a list of "word:count" format
        :param vocab: The vocabulary to use as features (list)
        """
        self.nonzero = {}
        self.y = label
        self.x = zeros(len(vocab))
        self.df = df
        for word, count in [x.split(":") for x in words]:
            if word in vocab:
                assert word != kBIAS, "Bias can't actually appear in document"
                self.x[vocab.index(word)] += float(count)
                self.nonzero[vocab.index(word)] = word
        self.x[0] = 1


class LogReg:
    def __init__(self, num_features, lam, eta=lambda x: 0.1, total_num_docs=0):
        """
        Create a logistic regression classifier

        :param num_features: The number of features (including bias)
        :param lam: Regularization parameter
        :param eta: A function that takes the iteration as an argument (the default is a constant value)
        """
        
        self.w = zeros(num_features)
        self.lam = lam
        # Comment out the following if you want to enable the scheduling function
        # self.eta = eta_schedule
        self.eta = eta
        self.last_update = defaultdict(int)
        self.total_num_docs = total_num_docs

        assert self.lam>= 0, "Regularization parameter must be non-negative"

    def progress(self, examples):
        """
        Given a set of examples, compute the probability and accuracy

        :param examples: The dataset to score
        :return: A tuple of (log probability, accuracy)
        """

        logprob = 0.0
        num_right = 0
        for ex in examples:
            p = sigmoid(self.w.dot(ex.x))
            if ex.y == 1:
                logprob += log(p)
            else:
                logprob += log(1.0 - p)

            # Get accuracy
            if abs(ex.y - p) < 0.5:
                num_right += 1

        return logprob, float(num_right) / float(len(examples))

    def sg_update(self, train_example, iteration, use_tfidf=False):
        """
        Compute a stochastic gradient update to improve the log likelihood.

        :param train_example: The example to take the gradient with respect to
        :param iteration: The current iteration (an integer)
        :param use_tfidf: A boolean to switch between the raw data and the tfidf representation
        :return: Return the new value of the regression coefficients
        """
        
        # TODO: Implement updates in this function
        if use_tfidf:
            tf_idf_x = [train_example.x[i] * log(self.total_num_docs / (1 + train_example.df[i])) for i in range(len(train_example.x))]
            muii = train_example.y - sigmoid(np.dot(self.w, tf_idf_x))
        else:
            muii = train_example.y - sigmoid(np.dot(self.w, train_example.x))

        self.w[0] += self.eta(iteration) * muii * train_example.x[0] # bias

        for kk in range(1, len(self.w)):
            if train_example.x[kk] != 0:
                if use_tfidf:
                    tf_idf_x = [train_example.x[i] * log(self.total_num_docs / (1 + train_example.df[i])) for i in range(len(train_example.x))]
                    self.w[kk] += self.eta(iteration) * (muii * tf_idf_x[kk])
                else:
                    self.w[kk] += self.eta(iteration) * (muii * train_example.x[kk])
                # Lazy Sparse Regularization, https://lingpipe.files.wordpress.com/2008/04/lazysgdregression.pdf
                self.w[kk] *= math.pow(1.0 - self.eta(iteration) * 2.0 * self.lam, iteration + 1 - self.last_update.get(kk, 0))
                self.last_update[kk] = iteration + 1

        return self.w

def read_dataset(positive, negative, vocab, test_proportion=0.1):
    """
    Reads in a text dataset with a given vocabulary

    :param positive: Positive examples
    :param negative: Negative examples
    :param vocab: A list of vocabulary words
    :param test_proprotion: How much of the data should be reserved for test
    """

    df = [float(x.split("\t")[1]) for x in open(vocab, 'r') if '\t' in x]
    vocab = [x.split("\t")[0] for x in open(vocab, 'r') if '\t' in x]
    assert vocab[0] == kBIAS, \
        "First vocab word must be bias term (was %s)" % vocab[0]

    train = []
    test = []
    for label, input in [(1, positive), (0, negative)]:
        for line in open(input):
            ex = Example(label, line.split(), vocab, df)
            if random.random() <= test_proportion:
                test.append(ex)
            else:
                train.append(ex)

    # Shuffle the data 
    random.shuffle(train)
    random.shuffle(test)

    return train, test, vocab

def eta_schedule(iteration):
    # TODO (extra credit): Update this function to provide an
    # EFFECTIVE iteration dependent learning rate size.  
    """
    0.1 / ((1.0*iteration / 100) + 1) is used to compute the graph shown in Figure 3 of the analysis report.
    However, in order not to break the unit tests, I've commented out the scheduling function.
    """
    #return 0.1 / ((1.0*iteration / 100) + 1)
    return 1.0 

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--lam", help="Weight of L2 regression",
                           type=float, default=0.0, required=False)
    argparser.add_argument("--eta", help="Initial SG learning rate",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--positive", help="Positive class",
                           type=str, default="../data/hockey_baseball/positive", required=False)
    argparser.add_argument("--negative", help="Negative class",
                           type=str, default="../data/hockey_baseball/negative", required=False)
    argparser.add_argument("--vocab", help="Vocabulary that can be features",
                           type=str, default="../data/hockey_baseball/vocab", required=False)
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=1, required=False)
    argparser.add_argument("--tfidf", help="Use tfidf as the feature",
                           type=bool, default=False, required=False)

    args = argparser.parse_args()
    train, test, vocab = read_dataset(args.positive, args.negative, args.vocab)

    print("Read in %i train and %i test" % (len(train), len(test)))

    # Logging file
    if args.passes != 1:
        log_file = open("learning_rate_%s_passes_%s.txt" % (args.eta, args.passes), "a")
    else:
        log_file = open("learning_rage_%s.txt" % args.eta, "a")
    
    # Initialize model
    lr = LogReg(len(vocab), args.lam, lambda x: args.eta, len(train))

    # Iterations
    iteration = 0
    for pp in xrange(args.passes):
        random.shuffle(train)
        for ex in train:
            lr.sg_update(ex, iteration, args.tfidf)
            if iteration % 5 == 1:
                train_lp, train_acc = lr.progress(train)
                ho_lp, ho_acc = lr.progress(test)
                print("Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
                      (iteration, train_lp, ho_lp, train_acc, ho_acc))
                # TODO: save the log to an external file
                log_file.write("Update %i\tTP %f\tHP %f\tTA %f\tHA %f\n" %
                      (iteration, train_lp, ho_lp, train_acc, ho_acc))
            iteration += 1

    # Printing out the words and the learned weights
    for k, v in sorted(zip(vocab, lr.w), key=lambda x:x[1], reverse=True):
        print(k, v)
