import cv2
import requests
import json
from hand_landmark_extractor import HandLandmarkExtractor
import time
import threading

class SignLanguageClient:
    def __init__(self, api_url="http://localhost:5000/predict"):
        self.api_url = api_url
        # Usar static_image_mode=False para melhor performance em vídeo
        self.extractor = HandLandmarkExtractor(static_image_mode=False, max_num_hands=1)
        self.last_prediction = "A aguardar..."
        self.last_confidence = 0.0
        self.processing = False

    def get_prediction(self, payload): # Corrigido: o nome do parâmetro deve ser payload
        """Função para ser executada numa thread separada para não bloquear o vídeo."""
        self.processing = True
        try:
            response = requests.post(
                self.api_url, 
                json=payload, # Agora a variável payload está definida como o argumento da função
                timeout=0.5
            )
            if response.status_code == 200:
                result = response.json()
                self.last_prediction = result['letter']
                self.last_confidence = result['confidence']
        except Exception as e:
            # Opcional: print(f"Erro na API: {e}")
            pass
        self.processing = False

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Iniciando captura otimizada... Pressione 'q' para sair.")
        
        frame_count = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            frame_count += 1
            
            # Processar apenas 1 em cada 5 frames para a IA
            if frame_count % 5 == 0 and not self.processing:
                # Extrai os dados da imagem
                hands_data = self.extractor.process_image_landmarks(image)
                
                if hands_data:
                    # Converte para DataFrame e depois para dicionário para envio JSON
                    df_hand = self.extractor.hands_data_to_dataframe([hands_data[0]])
                    payload = df_hand.iloc[0].to_dict()
                    
                    # Inicia a thread passando o payload como argumento
                    thread = threading.Thread(target=self.get_prediction, args=(payload,))
                    thread.start()
            
            # Desenhar o resultado
            text = f"Letra: {self.last_prediction} ({self.last_confidence:.2%})"
            cv2.rectangle(image, (5, 15), (350, 60), (0, 0, 0), -1)
            cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Detecao de Lingua Gestual (Otimizada)', image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    client = SignLanguageClient()
    client.run()