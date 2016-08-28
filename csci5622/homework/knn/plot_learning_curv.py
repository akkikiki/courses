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
    plt.xlim((0, 21))
    plt.xlabel("The value of k")
    train_sizes = range(1, 21) # size of the training data. len(train_sizes) = 19
    plt.ylabel("Accuracy")
    accuracy = []
    #for line in open("accuracies.txt"):
    #for line in open("learning_curve_accuracies.txt"):
    for line in open("k_accuracies.txt"):
        accuracy.append(float(line[:-1].split()[1]))
    print len(accuracy)
    assert len(accuracy) == len(train_sizes)
    plt.grid()
    plt.plot(train_sizes, accuracy, 'o-', color="r", label="Accuracy")

    plt.legend(loc="best")
    #plt.legend(loc=4)
    #plt.legend(loc=2)
    return plt

title = "Learning Curves"
plot_learning_curve(title)

plt.show()
