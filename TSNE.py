from sklearn import manifold, datasets
import csv
import numpy as np
import matplotlib.pyplot as plt
X = []
y = []
with open('feature.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        X.append(row[:-1])
        y.append(int(float(row[-1])))

X = np.array(X)
y = np.array(y)
print(y)
tsne = manifold.TSNE(n_components=2, init='pca', random_state=5, verbose=1)
X_tsne = tsne.fit_transform(X)
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
            fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()