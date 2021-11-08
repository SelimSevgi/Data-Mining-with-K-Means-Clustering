import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np

colormap = np.array(['red', 'blue', 'black'])
iris = datasets.load_iris()
x = pd.DataFrame(iris.data)
x.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
y = pd.DataFrame(iris.target) # Kümeleme
y.columns = ['Targets']

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.scatter(x.Sepal_Length, x.Sepal_Width, c=colormap[y.Targets], s=40)
plt.grid(color = 'green', linestyle = '--', linewidth = 0.3)
plt.title('Sepal')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.subplot(1, 2, 2)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Petal')
plt.grid(color = 'green', linestyle = '--', linewidth = 0.3)
plt.suptitle("Sepal and Petal Datas")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()
plt.close()

model = KMeans(algorithm="full", copy_x=True,init="k-means++",
               max_iter=300, n_clusters=3, n_init=10, random_state=0)
model.fit(x)

plt.figure(figsize=(14, 7))
colormap = np.array(['red', 'blue', 'black'])
print(model.cluster_centers_)

plt.subplot(1, 1, 1)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[model.labels_], s=40)
plt.scatter(model.cluster_centers_[:, 2], model.cluster_centers_[:, 3], c=colormap[:], s=150)
plt.grid(color = 'green', linestyle = '--', linewidth = 0.3)
plt.title('K Mean Classification')
plt.show()

predY = np.choose(model.labels_, [1, 0, 2]).astype(np.int64)
#print(predY)

print("Accuracy : ",sm.accuracy_score(y, predY))