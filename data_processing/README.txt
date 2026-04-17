- the dataset was too large to upload, even compressed. So i created a .npz that stores all the info about the label/unlabel data already processed.

-using this code on the main script, we should be able to "unpack" that file with all the infos. We just need to obviously have the .npz 
and the file with this same code (which i called "little_unpacking_script") in the same folder.


…hope it works


import numpy as np

data        = np.load("processed_data.npz")
A_train     = data["A_train"]
b_train     = data["b_train"]
A_test      = data["A_test"]
b_test      = data["b_test"]
X_unlabeled = data["X_unlabeled"]
k           = len(np.unique(b_train))