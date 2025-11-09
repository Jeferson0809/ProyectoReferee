<div align="center">

# ⚽ Clasificación de Eventos en Partidos de Fútbol  
### *Deep Learning aplicado a análisis de video deportivo*
<br>

[Jeferson Acevedo](https://github.com/Jeferson0809) • [Brayan Quintero](https://github.com/BrayanQuintero123) • [Reinaldo Cardenas](https://github.com/reinaldocardenas23)

---

</div>

El análisis automático de videos deportivos permite identificar y clasificar momentos relevantes dentro de un partido de fútbol —como **saques de banda, faltas, corners o tiros al arco**—, lo que facilita el análisis táctico, la indexación y la generación de resúmenes automáticos.

Este proyecto implementa un **sistema de clasificación de eventos futbolísticos** a partir de videos, utilizando **clips temporales generados automáticamente** y un **modelo 3D CNN** (por defecto `r3d_18` de TorchVision) entrenado sobre datos del conjunto **SoccerNet**.

> **Objetivo:** Detectar y clasificar eventos de fútbol a partir de clips cortos de video, con una interfaz visual desarrollada en Gradio.

---

## 🧠 Fundamento teórico

El enfoque está inspirado en el trabajo de Carreira & Zisserman (2017):  
**“Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset.”**  
📄 [Arxiv PDF](https://arxiv.org/pdf/1711.11248)

Dicho trabajo introdujo las **redes convolucionales 3D infladas (I3D)**, que extienden las convoluciones espaciales de 2D a 3D para capturar información temporal.  
Nuestro modelo sigue este principio, implementado mediante una **ResNet-3D (r3d_18)** de `torchvision`, optimizada para clips de fútbol.

---

## 🎯 Características principales

- 📹 **Entrada:** Clips de video cortos recortados directamente desde la interfaz.  
- 🧠 **Modelo base:** `ResNet-3D (r3d_18)` preentrenada en *Kinetics*, adaptada a 8 clases de eventos.  
- ⚡ **Procesamiento de video:**
  - Muestreo uniforme de `T = 16` frames.
  - Resize por lado corto (256 px) y *center crop* (112×112).
  - Normalización por canal (mean=0.45, std=0.225).  
- 💬 **Interfaz Gradio:**
  - Permite subir y recortar un clip.
  - Ejecuta la inferencia en tiempo real.
  - Genera un archivo `.py` descargable con la predicción.
- 📦 **Salida:**
  - Clase predicha y Top-3 probabilidades.
  - Archivo `.py` con código para imprimir la predicción.

---

## 🧩 Estructura del repositorio

```

Referee/
│
├── app.py               # Aplicación principal de Gradio
├── modelo.pth           # Pesos del modelo preentrenado
├── requirements.txt     # Dependencias del proyecto
│
├── data/                # (Opcional) Datasets o scripts de preparación
├── notebooks/           # Notebooks de entrenamiento / análisis
├── utils/               # Funciones auxiliares (lectura y procesado)
└── README.md            # Este archivo

````

---

## 🧠 Modelo y entrenamiento

### Arquitectura base
El modelo está basado en **ResNet-3D (r3d_18)** de `torchvision.models.video`, modificada para ajustarse al número de clases de SoccerNet:

```python
CLASS_LABELS = [
    "Ball out of play",
    "Throw-in",
    "Foul",
    "Indirect free-kick",
    "Clearance",
    "Shots on target",
    "Shots off target",
    "Corner"
]
````

### Proceso de entrenamiento

* Dataset: [**SoccerNet**](https://www.soccer-net.org/)
* Duración de clip: 16 frames (≈1.6 s a 10 FPS)
* División: 70% entrenamiento / 15% validación / 15% prueba
* Optimizador: `AdamW`
* Pérdida: `CrossEntropyLoss`
* Resolución: 112×112
* Regularización: *grad clip*, *label smoothing*, *mixup* (opcional)

---

## 🧪 Interfaz interactiva

App pública disponible en **Hugging Face Spaces** 👇
👉 [https://huggingface.co/spaces/Jeferson08/Referee](https://huggingface.co/spaces/Jeferson08/Referee)

**Funciones principales:**

1. Subir un clip de video (mp4, mkv, avi, etc.).
2. Recortarlo visualmente desde la interfaz (*Trim*).
3. Clasificar el evento con el modelo.
4. Descargar un `.py` con los resultados.

---

## 🎥 Video explicativo

[![Ver video en YouTube](https://img.youtube.com/vi/abcd1234xyz/hqdefault.jpg)](https://www.youtube.com/watch?v=abcd1234xyz)

---

## 📊 Ejemplo de salida

| Clase predicha  | Probabilidad | Ejemplo visual |
| --------------- | ------------ | -------------- |
| Shots on target | 0.91         | 🎯             |
| Foul            | 0.06         | 🚫             |
| Corner          | 0.02         | 🥅             |

> El modelo puede mejorarse con clips más largos o arquitecturas temporales (Transformer 3D, TimeSformer, etc.).

---

## 🧱 Tecnologías utilizadas

* **Python 3.10**
* **PyTorch / Torchvision**
* **OpenCV**
* **Gradio**
* **NumPy / Pandas**
* **SoccerNet Dataset**
---

<div align="center">

Hecho con ❤️ usando **PyTorch** y **Gradio**

</div>





