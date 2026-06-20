import wfdb
import numpy as np
import matplotlib.pyplot as plt

record = wfdb.rdrecord('100', pn_dir='mitdb/')
anotaciones = wfdb.rdann('100', 'atr', pn_dir='mitdb/')

señal = record.p_signal[:, 0]
#Creamos una ventana de 90 muestras antes y después del pico de cada latido
ventana = 90
latidos = []
etiquetas = []
#Creamos un bucle para recorrer todas las anotaciones y extraer los latidos correspondientes a cada pico
for i, simbolo in enumerate(anotaciones.symbol):
    if simbolo not in ['N', 'V', 'A']:
        continue
    
    pico = anotaciones.sample[i]
    
    if pico - ventana < 0 or pico + ventana > len(señal):
        continue
    
    recorte = señal[pico - ventana : pico + ventana]
    latidos.append(recorte)
    
    if simbolo == 'N':
        etiquetas.append(0)
    else:
        etiquetas.append(1)
latidos = np.array(latidos)
etiquetas = np.array(etiquetas)

print('Total latidos extraídos:', len(latidos))
print('Forma del array:', latidos.shape)
print('Latidos normales:', sum(etiquetas == 0))
print('Latidos con arritmia:', sum(etiquetas == 1))