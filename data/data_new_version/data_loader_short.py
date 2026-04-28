#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def load_blood_cells():
    data = np.load("blood_data.npz")
    return data["A_train"], data["b_train"], data["A_test"], data["b_test"]

