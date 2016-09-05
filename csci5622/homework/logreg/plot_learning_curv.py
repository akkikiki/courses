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

    #ylim=(0.70, 1.00)
    #if ylim is not None:
    #    plt.ylim(*ylim)

    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    iterations = []
    train_accuracies = []
    hold_out_accuracies = []
    #for line in open("learning_rage_0.1.txt"):
    for line in open("learning_rate_0.1_passes_2.txt"):
    #for line in open("learning_rage_0.2.txt"):
        # example Update 1061TP -284.858884HP -78.940187TA 0.946429HA 0.909774
        iteration, train_log_prob, hold_out_log_prob, train_accuracy, hold_out_accuracy = line[:-1].split("\t")
        train_accuracies.append(float(train_accuracy.replace("TA ", "")))
        hold_out_accuracies.append(float(hold_out_accuracy.replace("HA ", "")))
        iterations.append(int(iteration.replace("Update ", "")))
        
    print(iterations)
    print(train_accuracies)
    assert len(iterations) == len(train_accuracies)
    plt.grid()
    plt.plot(iterations, train_accuracies, 'o-', color="r", label="Training Accuracy")
    plt.plot(iterations, hold_out_accuracies, 'o-', color="g", label="Hold Out Accuracy")
    # TODO: Plot the held out accuracies for different learning rates

    plt.legend(loc="best")
    #plt.legend(loc=4)
    #plt.legend(loc=2)
    return plt

title = "Learning Curves"
plot_learning_curve(title)

plt.show()
