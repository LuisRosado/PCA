import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
import pickle

# Leer datos
mnist = fetch_openml('mnist_784')
data = mnist.data

# Definir número de ejemplos
n_samples = 10000  # Reducimos el número de muestras para acelerar el entrenamiento
x = np.asarray(data)[:n_samples, :]
y = np.asarray(mnist.target)[:n_samples].ravel()

# Dibujar un ejemplo de manera aleatoria
sample = np.random.randint(n_samples)
plt.imshow(x[sample].reshape((28, 28)), cmap=plt.cm.gray)
plt.title('Target: %s' % y[sample])
plt.show()

# Diferentes cantidades de componentes para PCA
n_components_list = [10, 20, 30, 40, 50]

# Entrenar y evaluar modelos para diferentes cantidades de componentes
for n_components in n_components_list:
    # Separar conjuntos de entrenamiento y prueba
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1, random_state=42)

    # Instanciar el Pipeline con PCA y SVM
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('PCA', PCA(n_components=n_components)),
        ('SVM', svm.SVC(gamma=0.0001))
    ])
    model.fit(xtrain, ytrain)

    # Aplicar métrica al modelo
    print(f'Número de Componentes: {n_components}')
    print('Train:', model.score(xtrain, ytrain))
    print('Test:', model.score(xtest, ytest))

    # Hacer predicciones del test
    ypred = model.predict(xtest)

    # Reporte de Clasificación
    print('Classification report:')
    print(metrics.classification_report(ytest, ypred))

    # Matriz de Confusión
    confusion_matrix = metrics.confusion_matrix(ytest, ypred)
    print('Confusion matrix:')
    print(confusion_matrix)

    # Guardar la matriz de confusión como .eps
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y)))
    plt.xticks(tick_marks, np.unique(y), rotation=45)
    plt.yticks(tick_marks, np.unique(y))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{n_components}components.eps', format='eps')
    plt.show()

    # Guardar modelo
    filename = f'Mnist_classifier_{n_components}components.sav'
    pickle.dump(model, open(filename, 'wb'))
