import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

Dataset = pd.read_csv("train.csv").sample(5000)
Y_df = Dataset['label']
X_df = Dataset.drop(['label'], axis=1)
X_df = X_df / 255.0
Y_np = Y_df.to_numpy()
X_np = X_df.to_numpy()
# Отрисовка изображения в matplotlib
X_np1 = X_df.values.reshape(-1, 28, 28, 1)
im = plt.imshow(X_np1[7][:,:,0])
plt.show()
tsne = TSNE()
X_tsne = tsne.fit_transform(X_np)
# Диаграма рассеяния обработанных данных
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
          "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]
plt.xlim(X_tsne[:, 0].min(), X_tsne[:, 0].max()+1)
plt.ylim(X_tsne[:, 1].min(), X_tsne[:, 1].max()+1)
for i in range(len(X_np)):
    plt.text(X_tsne[i, 0], X_tsne[i, 1],
             str(Y_np[i]),
             color=colors[Y_np[i]],
             fontdict={'weight': 'bold', 'size': 9})
plt.show()
