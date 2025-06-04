# Facial Emotion Recognition AI / Reconocimiento de Emociones Faciales con IA

## English

### Overview
This project implements a real-time facial emotion recognition system using deep learning. The system can detect and classify 7 different emotions: Anger, Disgust, Fear, Happy, Neutral, Sad, and Surprised.

### How it Works
The system uses a Convolutional Neural Network (CNN) trained on facial emotion data. [1](#0-0)  The model processes 48x48 grayscale images and outputs predictions for 7 emotion categories.

For real-time detection, the system: [2](#0-1) 
1. Captures video from webcam
2. Detects faces using Haar Cascade classifier
3. Extracts and preprocesses face regions
4. Predicts emotions using the trained model
5. Displays results in real-time

### Requirements
- Python 3.x
- TensorFlow/Keras
- OpenCV
- NumPy

### How to Run

#### Training the Model
```bash
python Facial-Emotion-Recognition-main/TrainEmotionDetector.py
```

#### Running Emotion Detection
```bash
python Facial-Emotion-Recognition-main/TestEmotionDetector.py
```
Press 'q' to quit the application.

---

## Español

### Descripción General
Este proyecto implementa un sistema de reconocimiento de emociones faciales en tiempo real usando aprendizaje profundo. El sistema puede detectar y clasificar 7 emociones diferentes: Enojo, Asco, Miedo, Feliz, Neutral, Triste y Sorprendido.

### Cómo Funciona
El sistema utiliza una Red Neuronal Convolucional (CNN) entrenada con datos de emociones faciales. [1](#0-0)  El modelo procesa imágenes en escala de grises de 48x48 píxeles y genera predicciones para 7 categorías de emociones.

Para la detección en tiempo real, el sistema: [2](#0-1) 
1. Captura video desde la cámara web
2. Detecta rostros usando el clasificador Haar Cascade
3. Extrae y preprocesa las regiones faciales
4. Predice emociones usando el modelo entrenado
5. Muestra resultados en tiempo real

### Requisitos
- Python 3.x
- TensorFlow/Keras
- OpenCV
- NumPy

### Cómo Ejecutar

#### Entrenar el Modelo
```bash
python Facial-Emotion-Recognition-main/TrainEmotionDetector.py
```

#### Ejecutar Detección de Emociones
```bash
python Facial-Emotion-Recognition-main/TestEmotionDetector.py
```
Presiona 'q' para salir de la aplicación.

## Notes

El sistema está configurado para detectar 7 emociones específicas según el diccionario definido en el código. [3](#0-2)  El modelo requiere datos de entrenamiento organizados en carpetas específicas y genera archivos de modelo (.json y .h5) que son utilizados por el sistema de detección en tiempo real.
