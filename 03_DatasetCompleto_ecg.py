import wfdb
import numpy as np

pacientes = ['100', '101', '104', '105', '106', '108', '109', '111']
ventana = 90
latidos = []
etiquetas = []

for paciente in pacientes:
    print(f'Cargando paciente {paciente}...')
    
    record = wfdb.rdrecord(paciente, pn_dir='mitdb/')
    anotaciones = wfdb.rdann(paciente, 'atr', pn_dir='mitdb/')
    señal = record.p_signal[:, 0]
    
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
print('Latidos normales:', sum(etiquetas == 0))
print('Latidos con arritmia:', sum(etiquetas == 1))

np.save('latidos.npy', latidos)
np.save('etiquetas.npy', etiquetas)

print('Dataset guardado!')