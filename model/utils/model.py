import imblearn
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.base import clone
from utils.plot import plot_learning_curve
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'Times New Roman'

models = {
    'svm': svm.SVC(C=1.0, kernel='rbf'),
    'dtree': DecisionTreeClassifier(splitter="random"),
    'balance_tree': imblearn.ensemble.EasyEnsembleClassifier(random_state=43, sampling_strategy='majority', n_estimators=7),
    'random_forest': RandomForestClassifier(),
    'bagging': BaggingClassifier(),
    'boost': AdaBoostClassifier()
}



def show_predict_result(y_true, y_pred):

    acc_score = accuracy_score(y_true, y_pred)
    print('accuracy score: {}'.format(acc_score))
    prec_score = precision_score(y_true, y_pred)
    print('precision score: {}'.format(prec_score))
    rec_score = recall_score(y_true, y_pred)
    print('recall score: {}'.format(rec_score))
    f_score = f1_score(y_true, y_pred)
    print('f1 score: {}'.format(f_score))
    return {'accuracy': acc_score,
            'precision': prec_score,
            'recall': rec_score,
            'f1': f_score}


def evaluate_model(name, X, Y, learning_curve=False):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    clf = models[name]
    curve_clf = clone(clf)
    clf.fit(x_train, y_train)
    print('train score: {}'.format(clf.score(x_train, y_train)))
    print('test score: {}'.format(clf.score(x_test, y_test)))
    y_pred = clf.predict(x_test)
    scores = show_predict_result(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)
    if learning_curve:
        plot_learning_curve(
            curve_clf,
            title="TCPSession Based RNN Classifier",
            X=x_train,
            y=y_train,
            cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=43),
            ylim=[0.92, 1.0])
    return clf, scores




# def model_svm(X, Y, learning_curve: bool = False):
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
#     clf = svm.SVC(C=1.0, kernel='rbf')
#     clf.fit(x_train, y_train)
#     train_score = clf.score(x_train, y_train)
#     print('train score: {}'.format(train_score))
#     test_score = clf.score(x_test, y_test)
#     print('test score: {}'.format(test_score))
#     y_pred = clf.predict(x_test)
#     show_predict_result(y_test, y_pred)
#
#     matrix = confusion_matrix(y_test, y_pred)
#     print(matrix)
#     if learning_curve:
#         plot_learning_curve(estimator=svm.SVC(C=1.0, kernel='rbf', gamma=0.5), title='svm model', X=X, y=Y,
#                             cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0))
#     return clf
#
#
# def model_balance_forest(X, Y, learning_curve: bool = False):
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
#     clf = imblearn.ensemble.EasyEnsembleClassifier(random_state=73, sampling_strategy='majority', n_estimators=7)
#     clf.fit(x_train, y_train)
#     print('train score: {}'.format(clf.score(x_train, y_train)))
#     print('test score: {}'.format(clf.score(x_test, y_test)))
#     y_pred = clf.predict(x_test)
#     show_predict_result(y_test, y_pred)
#     matrix = confusion_matrix(y_test, y_pred)
#     print(matrix)
#     if learning_curve:
#         plot_learning_curve(
#             imblearn.ensemble.EasyEnsembleClassifier(random_state=73, sampling_strategy='majority', n_estimators=7),
#             title='EasyEnsembleClassifier', X=X, y=Y,
#             cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=43),
#             ylim=[0.95, 1.0])
#     return clf
#
#
# def model_decision_tree(X, Y, learning_curve: bool = False):
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
#     clf = DecisionTreeClassifier(splitter="random")
#     clf.fit(x_train, y_train)
#     print('train score: {}'.format(clf.score(x_train, y_train)))
#     print('test score: {}'.format(clf.score(x_test, y_test)))
#     y_pred = clf.predict(x_test)
#     show_predict_result(y_test, y_pred)
#     matrix = confusion_matrix(y_test, y_pred)
#     print(matrix)
#     if learning_curve:
#         plot_learning_curve(
#             DecisionTreeClassifier(),
#             title="Decision Tree",
#             X=x_train,
#             y=y_train,
#             cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=43),
#             ylim=[0.98, 1.0])
#
#     return clf


# def compose_model(model1, model2, x_test, y_test):

