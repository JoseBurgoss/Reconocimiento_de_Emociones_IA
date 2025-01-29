import cv2
import numpy as np
from tensorflow.keras.models import model_from_json # type: ignore

emotion_dict = {0: "Enojo", 1: "Asco", 2: "Miedo", 3: "Feliz", 4: "Neutral", 5: "Triste", 6: "Sorprendido"}

# cargo el json y creo el modelo
with open('Facial-Emotion-Recognition-main/model/emotion_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
emotion_model = model_from_json(loaded_model_json)

# aqui le cargo el weight al modelo
emotion_model.load_weights("Facial-Emotion-Recognition-main/model/emotion_model.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))  
        
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
            
            # Extraigo el face ROI
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = cv2.resize(roi_gray_frame, (48, 48))
            cropped_img = np.expand_dims(np.expand_dims(cropped_img, axis=-1), axis=0)  # Add dimensions
            
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            
            # Display la prediccion de la emocion
            cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Emotion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:

    cap.release()
    cv2.destroyAllWindows()
