import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

latidos = np.load('latidos.npy')
etiquetas = np.load('etiquetas.npy')

print('Dataset cargado:', latidos.shape)
#Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    latidos, etiquetas, test_size=0.2, random_state=42
)

print('Entrenamiento:', X_train.shape)
print('Test:', X_test.shape)
#Entrenar un modelo de Random Forest
modelo = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

print('Entrenando modelo...')
modelo.fit(X_train, y_train)
print('Entrenamiento completado!')

predicciones = modelo.predict(X_test)
print(classification_report(y_test, predicciones, target_names=['Normal', 'Arritmia']))

joblib.dump(modelo, 'modelo_random_forest.pkl')
print('Modelo guardado!')