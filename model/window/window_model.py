import logging
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from sklearn import utils

from utils.model import evaluate_model
from utils.preprocessor import get_data
plt.rcParams['font.sans-serif'] = 'Times New Roman'

logging.basicConfig(level=logging.INFO)


def get_window_feature(df: DataFrame, window: int = 3):
    logging.info('parse window context feature from session df, windows = {}'.format(window))
    rows = df.values
    features = []
    for i in range(0, len(rows) - window + 1):
        arr = rows[i:i + window]
        features.append(arr.ravel())
    logging.info('feature count: {}'.format(len(features)))
    return features


def preprocess_window_features(window: int = 1):
    benign_df = get_data('data/benign/feature_window.csv')
    malicious_df = get_data('data/malicious/malicious_features_window.csv')
    benign_features = get_window_feature(benign_df, window)
    benign_label = np.zeros(shape=(1, len(benign_features)))
    # print(benign_features[0:2])
    malicious_features = get_window_feature(malicious_df, window)
    malicious_label = np.ones(shape=(1, len(malicious_features)))
    # print(malicious_features[0:2])
    total_features = benign_features + malicious_features
    total_labels = np.append(benign_label, malicious_label)
    X, Y = utils.shuffle(total_features, total_labels)
    return X, Y


def model_with_window():
    score_names = ['accuracy', 'recall', 'precision', 'f1']
    results = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
    for window in range(1, 6):
        X, Y = preprocess_window_features(window)
        print('evaluate model result with window value: {}'.format(window))
        clf, scores = evaluate_model('random_forest', X, Y, True)
        for name in score_names:
            results[name].append(scores[name])
    fig, axes = plt.subplots(figsize=(10, 5))
    x = list(range(1, 6))
    print(results)
    axes.plot(x, results['accuracy'], linewidth=1, color='blue', label='accuracy')
    axes.plot(x, results['recall'], linewidth=1, color='green', label='recall')
    axes.plot(x, results['precision'], linewidth=1, color='black', label='precision')
    axes.plot(x, results['f1'], linewidth=1, color='red', label='f1 score')
    axes.set_xlabel('window size')
    axes.set_title('Window Size vs. Index Curve')
    axes.set_ylim([0.98, 1.0])
    axes.grid(True)
    axes.legend(loc="best")


def generate_window_model(model_name="random_forest", window=1):
    X, Y = preprocess_window_features(window)
    clf, scores = evaluate_model(model_name, X, Y, True)
    out_dir = os.path.join('data/model', 'model')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_file = os.path.join(out_dir, '{}_model.blob'.format(model_name))
    joblib.dump(clf, out_file)


def main():
    model_with_window()
    plt.show()


if __name__ == '__main__':
    generate_window_model()
