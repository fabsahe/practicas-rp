import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# leer el dataset
df = pd.read_csv('cmc.csv')

# crear el conjunto de datos sin la columna target
X = df.drop(columns=['Class'])

# obtener el vector de clases
y = df['Class'].values

# dividir datos en entrenamiento y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1, stratify=y)

# Crear una instancia de knn
knn = KNeighborsClassifier(n_neighbors = 3)

# Entrenar el knn
knn.fit(X_train,y_train)

#knn.predict(X_test)[0:5]

# Comprobar la precision del knn con el conjunto de prueba
knn.score(X_test, y_test)

# Metodo leave one out
loo = LeaveOneOut()

# Crear una nueva instancia de knn (para kfold)
knn_kf = KNeighborsClassifier(n_neighbors=3)

# Crear una nueva instancia de knn (para leave one out)
knn_loo = KNeighborsClassifier(n_neighbors=3)

# validacion cruzada con k = 10 
kf_scores = cross_val_score(knn_kf, X, y, cv=10)

# validacion cruzada con leave one out
loo_scores =  cross_val_score(knn_loo, X, y, cv=loo)

# promedio de los resultados de validacion
# print(kf_scores)
# print('Promedio: {}'.format(np.mean(kf_scores)))

# Crear otras instancia de knn
knn2 = KNeighborsClassifier()
knn3 = KNeighborsClassifier()

# Crear vector de valores de k para el knn
param_grid = {'n_neighbors': np.arange(1, 25)}

# Usar gridsearch para probar todos los valores de k
knn_gskf = GridSearchCV(knn2, param_grid, cv=10)
knn_gsloo = GridSearchCV(knn3, param_grid, cv=loo)

# Entrenar nuevamente
knn_gskf.fit(X, y)

knn_gsloo.fit(X, y)

# Mostrar el mejor parametro para el knn
kbb = knn_gskf.best_params_
print("Cross-validation para KNN con k = 10")
print(kbb)

kloo = knn_gsloo.best_params_
print("Cross-validation para KNN con leave-one-out")
print(kloo)

# SVM
scaler = StandardScaler()
X2 = scaler.fit_transform(X)

C_range = np.logspace(-2, 2, 5)
gamma_range = np.logspace(-2, 2, 5)
param_grid_svm = dict(gamma=gamma_range, C=C_range)
grid_kf = GridSearchCV(SVC(), param_grid=param_grid_svm, cv=10)
grid_loo = GridSearchCV(SVC(), param_grid=param_grid_svm, cv=loo)
grid_kf.fit(X2, y)
grid_loo.fit(X2, y)

# Mostrar el mejor parametro para la svm
r1 =grid_kf.best_params_
print("Cross-validation para SVM con k = 10")
print(r1)

r2 = grid_loo.best_params_
print("Cross-validation para SVM con leave-one-out")
print(r2)
