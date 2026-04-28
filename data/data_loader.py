import os
import numpy as np
from PIL import Image

def load_blood_cells(dataset_dir="bloodcells_dataset", img_size=64):
    X, y = [], []
    classi = sorted([c for c in os.listdir(dataset_dir) 
                     if os.path.isdir(os.path.join(dataset_dir, c))])
    
    label_map = {classe: idx for idx, classe in enumerate(classi)}
    
    for idx, classe in enumerate(classi):
        class_folder = os.path.join(dataset_dir, classe)
        for filename in os.listdir(class_folder):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(class_folder, filename)
                img = Image.open(path).convert('L')
                img = img.resize((img_size, img_size))
                X.append(np.array(img).flatten() / 255.0)
                y.append(idx)
                
    return np.array(X), np.array(y), label_map