import cv2
import requests
from hand_landmark_extractor import HandLandmarkExtractor
import threading
import numpy as np

class SignLanguageClient:
    def __init__(self, api_url="http://localhost:5000/predict"):
        self.api_url = api_url
        self.extractor = HandLandmarkExtractor(static_image_mode=False, max_num_hands=1)
        
        self.session = requests.Session()
        
        self.last_prediction = "A esperar mao/API..."
        self.last_confidence = 0.0
        self.processing = False

    def get_prediction(self, payload):
        self.processing = True
        try:
            # self.session.post é mais rápido que requests.post
            response = self.session.post(self.api_url, json=payload, timeout=0.4)
            if response.status_code == 200:
                result = response.json()
                self.last_prediction = result['letter']
                self.last_confidence = result['confidence']
        except:
            pass
        self.processing = False

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Tentar aumentar o FPS da captura
        cap.set(cv2.CAP_PROP_FPS, 60)
        
        print("A iniciar cliente ... Clique 'q' para sair.")
        
        while cap.isOpened():
            success, image = cap.read()
            if not success: break
            
            image = cv2.flip(image, 1)
            h, w, _ = image.shape
            
            hands_data = self.extractor.process_image_landmarks(image)
            
            if hands_data:
                # Desenhar landmarks
                image = self.extractor.draw_landmarks(image, hands_data)

                # Zoom Tracking
                landmarks_px = np.array([[lm['x'] * w, lm['y'] * h] for lm in hands_data[0]['landmarks']])
                x_min, y_min = np.min(landmarks_px, axis=0)
                x_max, y_max = np.max(landmarks_px, axis=0)
                
                margin = 40
                x1, y1 = max(0, int(x_min - margin)), max(0, int(y_min - margin))
                x2, y2 = min(w, int(x_max + margin)), min(h, int(y_max + margin))
                
                if x2 > x1 and y2 > y1:
                    hand_crop = image[y1:y2, x1:x2].copy()
                    zoom_size = 160 
                    hand_zoom = cv2.resize(hand_crop, (zoom_size, zoom_size))
                    image[10:10+zoom_size, w-zoom_size-10:w-10] = hand_zoom
                    cv2.rectangle(image, (w-zoom_size-10, 10), (w-10, 10+zoom_size), (255, 255, 255), 2)

                # Criar payload manualmente
                if not self.processing:
                    hand = hands_data[0]
                    norm_lms = hand['landmarks_normalized']

                    payload = {'hand': hand['handedness']}
                    for i, name in enumerate(self.extractor.landmark_names):
                        payload[f"{name}_x"] = float(norm_lms[i, 0])
                        payload[f"{name}_y"] = float(norm_lms[i, 1])
                        payload[f"{name}_z"] = float(norm_lms[i, 2])
                    
                    thread = threading.Thread(target=self.get_prediction, args=(payload,), daemon=True)
                    thread.start()
            
            # Interface de utilizador
            text = f"{self.last_prediction} (Confianca: {self.last_confidence:.0%})"
            cv2.rectangle(image, (5, 15), (280, 65), (0, 0, 0), -1)
            cv2.putText(image, text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Fast Sign Language Detection', image)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
                
        cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()

if __name__ == "__main__":
    client = SignLanguageClient()
    client.run()