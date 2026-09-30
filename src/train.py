import os
import pickle
import yaml
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def load_data(use_augmented):
    if use_augmented:
        train = np.load('data/augmented/train_aug.npz')
    else:
        train = np.load('data/prepared/train.npz')
    X_train = train['X'].reshape(len(train['X']), -1)
    y_train = train['y']
    return X_train, y_train

def main():
    with open('params.yaml') as f:
        params = yaml.safe_load(f)
    train_params = params['train']
    model_name = train_params['model']
    use_augmented = train_params['use_augmented']
    seed = train_params['seed']

    X_train, y_train = load_data(use_augmented)

    if model_name == 'logreg':
        model = LogisticRegression(max_iter=train_params['logreg']['max_iter'], random_state=seed)
    elif model_name == 'rf':
        model = RandomForestClassifier(n_estimators=train_params['rf']['n_estimators'], random_state=seed, n_jobs=-1)
    elif model_name == 'mlp':
        model = MLPClassifier(hidden_layer_sizes=tuple(train_params['mlp']['hidden_layer_sizes']),
                              max_iter=train_params['mlp']['max_iter'], random_state=seed)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"Training {model_name} on {'augmented' if use_augmented else 'original'} data...")
    model.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved.")

if __name__ == '__main__':
    main()