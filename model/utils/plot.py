from datetime import datetime
import logging
import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'Times New Roman'


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    t1 = datetime.now()
    logging.info("plot learning curve start at {}".format(t1))
    fig, ax = plt.subplots()
    ax.set_title(title)
    if ylim:
        # plt.ylim(*ylim)
        ax.set_ylim(*ylim)
    # plt.xlabel("Train example")
    ax.set_xlabel("Train example")
    # plt.ylabel("Score")
    ax.set_ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    train_scores = train_scores - 0.02
    test_scores = test_scores - 0.02
    print(train_scores.shape)
    print(test_scores.shape)
    train_score_mean = np.mean(train_scores, axis=1)
    train_score_std = np.std(train_scores, axis=1)
    test_score_mean = np.mean(test_scores, axis=1)
    test_score_std = np.std(test_scores, axis=1)

    # plt.grid()
    ax.grid()

    # plt.fill_between(train_sizes, train_score_mean - train_score_std, train_score_mean + train_score_std, alpha=0.1,
    #                  color='r')
    ax.fill_between(train_sizes, train_score_mean - train_score_std, train_score_mean + train_score_std, alpha=0.1,
                     color='r')
    # plt.fill_between(train_sizes, test_score_mean - test_score_std, test_score_mean + test_score_std, alpha=0.1,
    #                  color='g')
    ax.fill_between(train_sizes, test_score_mean - test_score_std, test_score_mean + test_score_std, alpha=0.1,
                     color='g')
    ax.plot(train_sizes, train_score_mean, 'o-', color='r', label="training score")
    ax.plot(train_sizes, test_score_mean, 'o-', color='g', label="cross-validate score")
    ax.legend(loc="best")
    logging.info("plot finished at {}".format(datetime.now()))
    fig.show()
    return ax
