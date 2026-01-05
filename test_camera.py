import cv2
import os
import pandas as pd
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier # Para a classificação
from hand_landmark_extractor import HandLandmarkExtractor

# Configurações
DATASET_PATH = "SignAlphaSet"
CSV_FILE = "hand_landmarks_dataset.csv"

def run_gesture_recognition():
    """
    Opção 2: Carrega o CSV, treina um modelo rápido e identifica gestos via câmara.
    """
    if not os.path.exists(CSV_FILE):
        print(f"Erro: O ficheiro {CSV_FILE} não existe. Execute a Opção 1 primeiro.")
        return

    # 1. Carregar e Treinar o Classificador (k-NN)
    print("A carregar base de dados e a preparar modelo...")
    df = pd.read_csv(CSV_FILE)
    
    # As colunas de coordenadas começam após 'label' e 'hand'
    # Esperamos 63 colunas de coordenadas (21 pontos * 3 eixos)
    X = df.drop(columns=['label', 'hand', 'source_image'], errors='ignore')
    y = df['label']
    
    # Criamos o classificador (k=3 é um bom ponto de partida)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    print("Modelo pronto!")

    # 2. Iniciar Câmara
    extractor = HandLandmarkExtractor(static_image_mode=False, max_num_hands=1)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        hands_data = extractor.process_image_landmarks(frame)

        if hands_data:
            # 3. Extrair dados da mão atual (63 valores)
            hand = hands_data[0]
            flat_data = extractor.flatten_hand_data(hand)
            
            # Converter para o formato que o modelo espera (apenas as coordenadas)
            # Removemos 'hand' do dicionário para ficar apenas com os eixos x,y,z
            coords_only = {k: v for k, v in flat_data.items() if k != 'hand'}
            input_data = pd.DataFrame([coords_only])
            
            # 4. Predição
            predicted_letter = model.predict(input_data)[0]
            hand_side = hand['handedness'] # Esquerda ou Direita

            # 5. Visualização (Enunciado: Identificação da mão e Label)
            # Desenha landmarks
            frame = extractor.draw_landmarks(frame, hands_data)
            
            # Texto no ecrã
            color = (0, 255, 0) # Verde
            cv2.putText(frame, f"MAO: {hand_side}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"LETRA: {predicted_letter}", (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        cv2.imshow("Classificador de Gestos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

if __name__ == "__main__":
    print("1. Gerar Dataset (Extrair Landmarks)")
    print("2. Identificar Gestos (Tempo Real)")
    choice = input("Opção: ")
    
    if choice == '1':
        # Chame a sua função de gerar dataset aqui
        pass 
    elif choice == '2':
        run_gesture_recognition()