import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html


def plot_learning_curve(title):
    """
    Generate a simple plot of the test and traning learning curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects
    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure(figsize=(10,5))
    matplotlib.rcParams.update({'font.size': 16})

    # plt.title(title)
    ylim=(0.70, 1.00)
    if ylim is not None:
        plt.ylim(*ylim)
    #plt.xlim((0, 10500))
    #plt.xlabel("Number of training data")
    #train_sizes = range(500, 10500, 500) # size of the training data. len(train_sizes) = 19
    plt.xlabel("Boosting Iteration")
    plt.ylabel("Accuracy")
    accuracy_training = []
    accuracy_testing = []

    for line in open("learners_500_max_depth_1_training"):
        accuracy_training.append(float(line[:-1]))
    for line in open("learners_500_max_depth_1_testing"):
        accuracy_testing.append(float(line[:-1]))
 
    print len(accuracy_training)
    train_sizes = range(len(accuracy_training)) # size of the training data. len(train_sizes) = 19
    assert len(accuracy_training) == len(train_sizes)
    plt.xlim((0, len(accuracy_training) + 1))
    plt.grid()
    plt.plot(train_sizes, accuracy_training, 'o-', color="r", label="Train Accuracy")
    plt.plot(train_sizes, accuracy_testing, '*-', color="b", label="Test Accuracy")

    plt.legend(loc="best")
    #plt.legend(loc=4)
    #plt.legend(loc=2)
    return plt

title = "Learning Curves"
plot_learning_curve(title)

plt.show()
