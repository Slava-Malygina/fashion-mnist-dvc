import os
import numpy as np
import yaml
from torchvision import datasets

def main():
    with open('params.yaml') as f:
        params = yaml.safe_load(f)
    seed = params['prepare']['seed']
    np.random.seed(seed)

    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/prepared', exist_ok=True)

    train = datasets.FashionMNIST(root='data/raw', train=True, download=True)
    test = datasets.FashionMNIST(root='data/raw', train=False, download=True)

    X_train = train.data.numpy().astype(np.float32) / 255.0
    y_train = train.targets.numpy().astype(np.int64)
    X_test = test.data.numpy().astype(np.float32) / 255.0
    y_test = test.targets.numpy().astype(np.int64)

    np.savez_compressed('data/prepared/train.npz', X=X_train, y=y_train)
    np.savez_compressed('data/prepared/test.npz', X=X_test, y=y_test)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

if __name__ == '__main__':
    main()