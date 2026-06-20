import wfdb
#Importamos la libreria y leemos el registro 100
record = wfdb.rdrecord('100', pn_dir='mitdb/')
anotaciones = wfdb.rdann('100', 'atr', pn_dir='mitdb/')
print('Símbolos únicos:', set(anotaciones.symbol))
print('Total anotaciones:', len(anotaciones.symbol))
print('Frecuencia de muestreo:', record.fs)
print('Nombre de las señales:', record.sig_name)
print('Duración en muestras:', record.sig_len)
print('Duración en segundos:', record.sig_len / record.fs)

#Importamos la libreria para graficar y vemos la señal del primer canal (MLII) 
import matplotlib.pyplot as plt

señal = record.p_signal[:1800, 0]
tiempo = [i / record.fs for i in range(len(señal))]

plt.figure(figsize=(12, 4))
plt.plot(tiempo,señal, color='blue', alpha=0.7)
#Creamos un bucle para graficar los picos de cada latido y los clasificamos por color según el tipo de latido
for i, simbolo in enumerate(anotaciones.symbol):
    posicion = anotaciones.sample[i]
    if posicion < 1800:
        t = posicion / record.fs
        if simbolo == 'N':
            plt.scatter(t, señal[posicion], color='green', s=50)
        elif simbolo == 'A':
            plt.scatter(t, señal[posicion], color='red', s=50)
        elif simbolo == 'V':
            plt.scatter(t, señal[posicion], color='orange', s=50)
plt.title('ECG - Paciente 100')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (mV)')
plt.legend(['Señal', 'Normal', 'Arritmia auricular', 'Arritmia ventricular'])
plt.show()
