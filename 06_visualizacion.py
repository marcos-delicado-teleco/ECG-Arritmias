import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
from tensorflow import keras

latidos = np.load('latidos.npy')
etiquetas = np.load('etiquetas.npy')

X_train, X_test, y_train, y_test = train_test_split(
    latidos, etiquetas, test_size=0.2, random_state=42
)

rf = joblib.load('modelo_random_forest.pkl')
rn = keras.models.load_model('modelo_red_neuronal.keras')

print('Modelos cargados!')

pred_rf = rf.predict(X_test)
pred_rn = (rn.predict(X_test) > 0.5).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_rf = confusion_matrix(y_test, pred_rf)
disp_rf = ConfusionMatrixDisplay(cm_rf, display_labels=['Normal', 'Arritmia'])
disp_rf.plot(ax=axes[0], colorbar=False)
axes[0].set_title('Random Forest')

cm_rn = confusion_matrix(y_test, pred_rn)
disp_rn = ConfusionMatrixDisplay(cm_rn, display_labels=['Normal', 'Arritmia'])
disp_rn.plot(ax=axes[1], colorbar=False)
axes[1].set_title('Red Neuronal')

plt.suptitle('Comparativa de modelos - Detección de Arritmias', fontsize=14)
plt.tight_layout()
plt.savefig('comparativa_modelos.png', dpi=150)

accuracy = np.load('historial_accuracy.npy')
val_accuracy = np.load('historial_val_accuracy.npy')

plt.figure(figsize=(10, 5))
plt.plot(accuracy, label='Entrenamiento')
plt.plot(val_accuracy, label='Validación')
plt.title('Curva de aprendizaje - Red Neuronal')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('curva_aprendizaje.png', dpi=150)
plt.show()