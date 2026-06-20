import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras import layers

latidos = np.load('latidos.npy')
etiquetas = np.load('etiquetas.npy')

X_train, X_test, y_train, y_test = train_test_split(
    latidos, etiquetas, test_size=0.2, random_state=42
)

print('Datos cargados y divididos')

modelo = keras.Sequential([
    layers.Input(shape=(180,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

modelo.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

modelo.summary()

historial = modelo.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    class_weight={0: 1, 1: 15}
)

predicciones = (modelo.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, predicciones, target_names=['Normal', 'Arritmia']))

np.save('historial_accuracy.npy', historial.history['accuracy'])
np.save('historial_val_accuracy.npy', historial.history['val_accuracy'])
modelo.save('modelo_red_neuronal.keras')
print('Red neuronal guardada!')