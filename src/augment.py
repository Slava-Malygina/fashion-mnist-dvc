import os
import numpy as np
import yaml
import random
from torchvision import transforms

def get_transform(rotation, shift, zoom):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=rotation),
        transforms.RandomAffine(degrees=0,
                                translate=(shift, shift),
                                scale=(1 - zoom, 1 + zoom)),
        transforms.ToTensor()
    ])

def main():
    with open('params.yaml') as f:
        params = yaml.safe_load(f)
    aug = params['augment']
    seed = aug['seed']
    random.seed(seed)
    np.random.seed(seed)

    train = np.load('data/prepared/train.npz')
    X_train, y_train = train['X'], train['y']
    N = len(X_train)
    n_augment = aug['n_augment']

    transform = get_transform(aug['rotation'], aug['shift'], aug['zoom'])

    X_list = []
    y_list = []
    for i in range(N):
        img = X_train[i]
        label = int(y_train[i])
        X_list.append(img)
        y_list.append(label)

        for _ in range(n_augment):
            aug_tensor = transform(img)
            aug_img = aug_tensor.squeeze(0).numpy()
            X_list.append(aug_img)
            y_list.append(label)

    X_aug = np.stack(X_list).astype(np.float32)
    y_aug = np.array(y_list).astype(np.int64)

    os.makedirs('data/augmented', exist_ok=True)
    np.savez_compressed('data/augmented/train_aug.npz', X=X_aug, y=y_aug)
    print(f"Augmented train: {X_aug.shape}")

if __name__ == '__main__':
    main()