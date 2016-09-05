import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

def read_data(filename):
    iterations = []
    train_accuracies = []
    hold_out_accuracies = []
    for line in open(filename):
        # example Update 1061TP -284.858884HP -78.940187TA 0.946429HA 0.909774
        iteration, train_log_prob, hold_out_log_prob, train_accuracy, hold_out_accuracy = line[:-1].split("\t")
        train_accuracies.append(float(train_accuracy.replace("TA ", "")))
        hold_out_accuracies.append(float(hold_out_accuracy.replace("HA ", "")))
        iterations.append(int(iteration.replace("Update ", "")))
        print train_accuracy
    return iterations, train_accuracies, hold_out_accuracies

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
    plt.ylabel("Hold out Accuracy")
    iterations = []
    train_accuracies = []
    hold_out_accuracies = []
    iterations, train_accuracies_0_1, hold_out_accuracies_0_1 = read_data("learning_rage_0.1.txt")
    _, train_accuracies_0_01, hold_out_accuracies_0_01 = read_data("learning_rage_0.01.txt")
    _, train_accuracies_1_0, hold_out_accuracies_1_0 = read_data("learning_rage_1.0.txt")
    _, train_accuracies_10_0, hold_out_accuracies_10_0 = read_data("learning_rage_10.0.txt")
    #_, train_accuracies_100_0 = read_data("learning_rage_100.0.txt")
    iterations, _, hold_out_accuracies_0_1_scheduled = read_data("learning_rage_0.1_scheduled.txt")
    iterations, _, hold_out_accuracies_0_1_not_scheduled = read_data("learning_rage_0.1_not_scheduled.txt")
        
    print(iterations)
    assert len(iterations) == len(train_accuracies_0_1)
    plt.grid()
    #plt.plot(iterations, train_accuracies_0_1, 'o-', color="r", label="Learning rate = 0.1")
    #plt.plot(iterations, train_accuracies_1_0, 'o-', color="g", label="Learning rate = 1.0")
    #plt.plot(iterations, train_accuracies_10_0, 'o-', color="b", label="Learning rate = 10.0")
    # TODO: Plot the held out accuracies for different learning rates
    plt.plot(iterations, hold_out_accuracies_0_01, 'o-', color="y", label="Learning rate = 0.01")
    plt.plot(iterations, hold_out_accuracies_0_1, 'o-', color="r", label="Learning rate = 0.1")
    plt.plot(iterations, hold_out_accuracies_1_0, 'o-', color="g", label="Learning rate = 1.0")
    plt.plot(iterations, hold_out_accuracies_10_0, 'o-', color="b", label="Learning rate = 10.0")

    #plt.plot(iterations, hold_out_accuracies_0_1_scheduled, 'o-', color="y", label="Learning rate = 0.1 / ((1.0*iteration / 100) + 1)")
    #plt.plot(iterations, hold_out_accuracies_0_1_not_scheduled, 'o-', color="r", label="Learning rate = 0.1")
 

 

    #plt.legend(loc="best")
    plt.legend(loc=4)
    #plt.legend(loc=2)
    return plt

title = "Learning Curves"
plot_learning_curve(title)

plt.show()
