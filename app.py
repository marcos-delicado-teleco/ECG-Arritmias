import streamlit as st
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from tensorflow import keras
import joblib

st.title('🫀 Detector de Arritmias ECG')
st.write('Selecciona un paciente del MIT-BIH Arrhythmia Database y analiza su señal ECG con IA')

modelo = keras.models.load_model('modelo_red_neuronal.keras')
rf_modelo = joblib.load('modelo_random_forest.pkl')

pacientes_disponibles = ['100', '101', '104', '105', '106', '108', '109', '111']
paciente = st.selectbox('Selecciona un paciente', pacientes_disponibles)

st.subheader(f'Señal ECG - Paciente {paciente}')

with st.spinner('Cargando señal ECG...'):
    record = wfdb.rdrecord(paciente, pn_dir='mitdb/')
    anotaciones = wfdb.rdann(paciente, 'atr', pn_dir='mitdb/')
    señal = record.p_signal[:1800, 0]
    tiempo = [i / record.fs for i in range(len(señal))]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(tiempo, señal, color='steelblue', alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Amplitud (mV)')
    ax.legend(['Señal ECG'])
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
st.subheader('Análisis con IA')

col1, col2 = st.columns(2)
boton_rn = col1.button('🧠 Analizar con Red Neuronal')
boton_rf = col2.button('🌲 Analizar con Random Forest')

if boton_rn or boton_rf:
    with st.spinner('Analizando...'):
        señal_completa = record.p_signal[:, 0]
        ventana = 90
        latidos = []
        posiciones = []

        for i, simbolo in enumerate(anotaciones.symbol):
            if simbolo not in ['N', 'V', 'A', 'L', 'R', 'B', 'a', 'J', 'S', 'e', 'j']:
                continue
            pico = anotaciones.sample[i]
            if pico - ventana < 0 or pico + ventana > len(señal_completa):
                continue
            latidos.append(señal_completa[pico - ventana : pico + ventana])
            posiciones.append(pico)

        latidos = np.array(latidos)
        if boton_rn:
            st.write('🧠 Usando Red Neuronal')
            predicciones = (modelo.predict(latidos) > 0.5).astype(int)
        else:
            st.write('🌲 Usando Random Forest')
            predicciones = rf_modelo.predict(latidos)

        normales = sum(predicciones == 0)
        arritmias = sum(predicciones == 1)

        st.metric('Total latidos analizados', len(predicciones))
        st.metric('Latidos normales ✅', int(normales.item()))
        st.metric('Arritmias detectadas ⚠️', int(arritmias.item()))
        st.write(f'Anotaciones encontradas: {len(anotaciones.symbol)}')
        st.write(f'Símbolos únicos: {set(anotaciones.symbol)}')

        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.pie(
            [int(normales.item()), int(arritmias.item())],
            labels=['Normal', 'Arritmia'],
            colors=['#2ecc71', '#e74c3c'],
            autopct='%1.1f%%'
        )
        ax2.set_title('Distribución de latidos')
        st.pyplot(fig2)

st.divider()
st.caption('Datos: MIT-BIH Arrhythmia Database | Modelos: Random Forest y Red Neuronal | Desarrollado por Marcos Delicado García')