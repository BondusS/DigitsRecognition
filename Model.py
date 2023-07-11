import pandas as pd
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

Dataset = pd.read_csv("train.csv").sample(5000)
Y_df = Dataset['label']
X_df = Dataset.drop(['label'], axis=1)
X_df = X_df / 255.0
Y_np = Y_df.to_numpy()
X_np = X_df.to_numpy()
tsne = TSNE()
X_tsne = tsne.fit_transform(X_np)
x_train, x_test, y_train, y_test = train_test_split(X_tsne, Y_np)
Model = KNeighborsClassifier().fit(x_train, y_train)
print('Доля правильных ответов на обучающей выборке ', Model.score(x_train, y_train))
print('Доля правильных ответов на тестовой выборке ', Model.score(x_test, y_test))
print('Подборка лучших параметров...')
params = {'n_neighbors': [i for i in range(1, 11)]}
grid = GridSearchCV(estimator=Model, param_grid=params)
grid.fit(x_train, y_train)
print('Наилучшая доля при подборке наиболее подходящих параметров ', grid.best_score_)
print('Наиболее подходящее количество соседей', grid.best_estimator_.n_neighbors)
BestModel = KNeighborsClassifier(n_neighbors=grid.best_estimator_.n_neighbors).fit(x_train, y_train)
print('Доля правильных ответов на обучающей выборке ', BestModel.score(x_train, y_train))
print('Доля правильных ответов на тестовой выборке ', BestModel.score(x_test, y_test))
