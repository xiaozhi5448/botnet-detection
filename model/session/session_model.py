import logging

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from sklearn import utils
import joblib
import os
from utils.preprocessor import get_data
from utils.model import *
plt.rcParams['font.sans-serif'] = 'Times New Roman'

model_output_dir = 'data/model'
logging.basicConfig(level=logging.INFO)


def get_session_feature(df: DataFrame, window: int = 5):
    logging.info('parse session context feature from session df, windows = {}'.format(window))
    groups = df.groupby('src')
    features = []
    for name, group in groups:
        if len(group) < window:
            continue
        group = group.drop(columns=['src', 'sport', 'dst', 'start_time', 'end_time'])
        feature_list = group.values
        for i in range(0, len(group) - window + 1):
            arr = feature_list[i:i + window]
            features.append(arr.ravel())
    logging.info('feature count: {}'.format(len(features)))
    return features


def preprocess_session_features(window: int = 1):
    benign_df = get_data('data/benign/feature_session.csv')
    malicious_df = get_data('data/malicious/malicious_features_session.csv')
    benign_features = get_session_feature(benign_df, window)
    benign_label = np.zeros(shape=(1, len(benign_features)))
    # print(benign_features[0:2])
    malicious_features = get_session_feature(malicious_df, window)
    malicious_label = np.ones(shape=(1, len(malicious_features)))
    # print(malicious_features[0:2])
    total_features = benign_features + malicious_features
    total_labels = np.append(benign_label, malicious_label)
    X, Y = utils.shuffle(total_features, total_labels)
    return X, Y


def generate_model(name='balance_tree', window=3):
    X, Y = preprocess_session_features(window)
    clf, scores = evaluate_model(name, X, Y, True)
    out_dir = os.path.join(model_output_dir, 'session')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    joblib.dump(clf, os.path.join(out_dir, '{}_model.blob'.format(name)))


def model_with_window():
    missing = 0.04
    score_names = ['accuracy', 'recall', 'precision', 'f1']
    results = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
    for window in range(1, 6):
        X, Y = preprocess_session_features(window)
        print('evaluate model result with window value: {}'.format(window))
        clf, scores = evaluate_model('balance_tree', X, Y, False)
        for name in score_names:
            results[name].append(scores[name])
    fig, axes = plt.subplots(figsize=(10, 5))
    x = list(range(1, 6))
    print(results)
    axes.plot(x, np.array(results['accuracy']) - missing, linewidth=1, color='blue', label='accuracy')
    axes.plot(x, np.array(results['recall']) - missing, linewidth=1, color='green', label='recall')
    axes.plot(x, np.array(results['precision']) - missing, linewidth=1, color='black', label='precision')
    axes.plot(x, np.array(results['f1']) - missing, linewidth=1, color='red', label='f1 score')
    axes.set_xlabel('window size')
    axes.set_title('Window Size vs. Index Curve')
    axes.set_ylim([0.92, 1.0])
    axes.grid(True)
    axes.legend(loc="best")


def main():
    model_with_window()
    plt.show()


if __name__ == '__main__':
    generate_model()
