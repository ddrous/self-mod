#%%
from selfmod import TrendsDataset
import numpy as np
import matplotlib.pyplot as plt


data_folder = "./data/"
skip_steps = 1
train_proportion = 1.0

train_dataset = TrendsDataset(data_dir=data_folder, 
                                skip_steps=skip_steps, 
                                traj_prop_min=train_proportion)

run_folder = "./tmp/"


#%%
X = train_dataset.dataset.squeeze()

## Normalize the data
# X = (X - X.mean(axis=0)) / X.std(axis=0)


print(X.shape)

# 1-100   Normal
# 101-200 Cyclic
# 201-300 Increasing trend
# 301-400 Decreasing trend
# 401-500 Upward shift
# 501-600 Downward shift

## We have 600 samples and 6 classes as above. Create the labels
labels = np.zeros((600,), dtype=int)
labels[100:200] = 1 
labels[200:300] = 2
labels[300:400] = 3
labels[400:500] = 4
labels[500:600] = 5

color_table = {0:"royalblue", 1:"crimson", 2:"forestgreen", 3:"darkorange", 4:"purple", 5:"black"}
colors = [color_table[l] for l in labels]

conditions = {0:"Normal", 1:"Cyclic", 2:"Increasing trend", 3:"Decreasing trend", 4:"Upward shift", 5:"Downward shift"}

## Use PCA instead
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
# X_reduced = X

# Plotting
plt.figure(figsize=(10, 7))
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], s=50, c=colors)

markers = {0:'o', 1:'x', 2:'^', 3:'s', 4:'D', 5:'P'}
for class_label in range(6):
    marker = markers[class_label]
    plt.scatter(X_reduced[labels==class_label, 0], X_reduced[labels==class_label, 1], s=50, c=color_table[class_label], label=conditions[class_label], marker=marker)

plt.legend()

plt.title("PCA Contexts Clustering", fontsize=24)
plt.xlabel("PC 1")
plt.ylabel("PC 2")

plt.draw()
plt.savefig(run_folder+"pc_contexts.png", bbox_inches='tight');




#%%

## Use Umap instead
import umap
reducer = umap.UMAP(n_components=2)
X_reduced = reducer.fit_transform(X)

# Plotting
plt.figure(figsize=(10, 7))

markers = {0:'o', 1:'x', 2:'^', 3:'s', 4:'D', 5:'P'}
for class_label in range(6):
    marker = markers[class_label]
    plt.scatter(X_reduced[labels==class_label, 0], X_reduced[labels==class_label, 1], s=50, c=color_table[class_label], label=conditions[class_label], marker=marker)

plt.legend()

plt.title("UMAP Contexts Clustering", fontsize=24)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")

plt.draw()
plt.savefig(run_folder+"umap_contexts.png", bbox_inches='tight');

