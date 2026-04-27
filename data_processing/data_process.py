import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter

# GLOBAL PARAMETERS

IMG_SIZE     = 64
N_COMPONENTS = 150
RANDOM_STATE = 42

LABELED_RATIO   = 0.20
UNLABELED_RATIO = 0.70
TEST_RATIO      = 0.10

DATASET_DIR = "bloodcells_dataset"


# READING FROM FOLDERS

def load_images(folder, has_labels=True):
    X, y, label_map = [], [], {}
    uploaded = 0

    if has_labels:
        classi = sorted([c for c in os.listdir(folder)
                         if os.path.isdir(os.path.join(folder, c))])
        
        for idx, classe in enumerate(classi):
            label_map[classe] = idx
            class_folder = os.path.join(folder, classe)
            
            for filename in os.listdir(class_folder):
                if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                    
                path = os.path.join(class_folder, filename)
                img  = Image.open(path).convert('L')
                img  = img.resize((IMG_SIZE, IMG_SIZE))
                
                vettore = np.array(img).flatten() / 255.0
                X.append(vettore)
                y.append(idx)
                uploaded += 1
    else:
        for filename in os.listdir(folder):
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            path = os.path.join(folder, filename)
            img  = Image.open(path).convert('L')
            img  = img.resize((IMG_SIZE, IMG_SIZE))
            vettore = np.array(img).flatten() / 255.0
            X.append(vettore)
            uploaded += 1

    print(f"Total images found: {uploaded}")

    if has_labels:
        return np.array(X), np.array(y), label_map
    else:
        return np.array(X), None, None

# INFO AND UPLOAD

X_all, y_all, label_map = load_images(DATASET_DIR, has_labels=True)

print(f"\nNumber of used images: {X_all.shape[0]}")
print(f"\nName and index of classes: {label_map}")

inverse_map = {v: k for k, v in label_map.items()}
distribution = {inverse_map[k]: v for k, v in Counter(y_all).items()}
print(f"Elements per class:  {distribution}")




# SPLIT

print("\nSplit labeled / unlabeled / test")

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_all, y_all,
    test_size=TEST_RATIO,
    stratify=y_all,
    random_state=RANDOM_STATE
)

labeled_size = LABELED_RATIO / (LABELED_RATIO + UNLABELED_RATIO)

X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
    X_trainval, y_trainval,
    train_size=labeled_size,
    stratify=y_trainval,
    random_state=RANDOM_STATE
)

print(f"Labeled:   {X_labeled.shape[0]:>6} ({LABELED_RATIO*100:.0f}%)")
print(f"Unlabeled: {X_unlabeled.shape[0]:>6}  ({UNLABELED_RATIO*100:.0f}%)")
print(f"Test:      {X_test.shape[0]:>6} ({TEST_RATIO*100:.0f}%)")


# STRANDARDIZATION

scaler = StandardScaler()
scaler.fit(X_labeled)

X_labeled_s   = scaler.transform(X_labeled)
X_unlabeled_s = scaler.transform(X_unlabeled)
X_test_s      = scaler.transform(X_test)



# PCA


pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
pca.fit(X_labeled_s)

X_labeled_pca   = pca.transform(X_labeled_s)
X_unlabeled_pca = pca.transform(X_unlabeled_s)
X_test_pca      = pca.transform(X_test_s)

variance = np.sum(pca.explained_variance_ratio_) * 100
print(f"\nPCA components: {N_COMPONENTS}")
print(f"Variance (useful data info kept after): {variance:.1f}%")

#with 50 components, we had 55% variance
#with 100, we have 68.8%
#with 150, we have 76%
#with 200, we have 81,6%

#FINAL FIT


A_train = X_labeled_pca
b_train = y_labeled
A_test  = X_test_pca
b_test  = y_test
k       = len(label_map)

print(f"A_train:     {A_train.shape}   for GD / BCGD")
print(f"b_train:     {b_train.shape}")
print(f"A_test:      {A_test.shape}    for final test")
print(f"b_test:      {b_test.shape}")
print(f"X_unlabeled: {X_unlabeled_pca.shape}")

# NPZ

np.savez_compressed(
    "processed_data.npz",
    A_train     = A_train,
    b_train     = b_train,
    A_test      = A_test,
    b_test      = b_test,
    X_unlabeled = X_unlabeled_pca
)

