import json
import os
import pickle
import yaml
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def main():
    with open('params.yaml') as f:
        params = yaml.safe_load(f)

    test = np.load('data/prepared/test.npz')
    X_test = test['X'].reshape(len(test['X']), -1)
    y_test = test['y']

    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    os.makedirs('eval', exist_ok=True)
    with open('eval/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Accuracy: {metrics['accuracy']:.4f}, F1 macro: {metrics['f1_macro']:.4f}")

if __name__ == '__main__':
    main()