"""
Ferramentas para processamento de gestos:
1. Geração de Dataset (Batch processing de SignAlphaSet)
2. Teste mínimo de câmera (Visualização em tempo real)

Este script permite iterar sobre as pastas do dataset, extrair os landmarks
e salvar um CSV consolidado com Label e Coordenadas.
"""

import cv2
import os
import glob
import pandas as pd
import time
from pprint import pprint
from typing import Optional
from hand_landmark_extractor import HandLandmarkExtractor

# Configurações do Dataset
DATASET_PATH = "SignAlphaSet"  # Pasta raiz contendo subpastas A, B, C...
OUTPUT_DATASET_FILE = "hand_landmarks_dataset.csv"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Configurações da Câmera
WINDOW_NAME = "Hand Landmarks"
CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def generate_dataset_from_folder(root_folder: str = DATASET_PATH, output_file: str = OUTPUT_DATASET_FILE):
    """
    Itera sobre todas as pastas do dataset (A-Z), processa as imagens para extrair landmarks
    e salva um dataset estruturado em CSV.
    """
    print(f"--- A iniciar geração do dataset a partir de: {root_folder} ---")
    
    if not os.path.exists(root_folder):
        print(f"Erro: A pasta '{root_folder}' não foi encontrada.")
        return

    # Inicializa extrator em modo estático (maior precisão para imagens soltas)
    extractor = HandLandmarkExtractor(
        static_image_mode=True,
        max_num_hands=1,  # Geralmente é 1 mão por letra no dataset, ajustar se necessário
        min_detection_confidence=0.5
    )

    dataset_rows = []
    total_images = 0
    processed_images = 0

    # Listar subpastas (que representam as labels/letras)
    subfolders = sorted([d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))])

    for label in subfolders:
        folder_path = os.path.join(root_folder, label)
        print(f"Processando pasta/label: {label}...")
        
        # Encontrar todas as imagens na pasta
        image_files = []
        for ext in ALLOWED_EXTENSIONS:
            # Case insensitive search se possível, ou apenas extensões minúsculas
            image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
        
        # Remove duplicados se houver e ordena
        image_files = sorted(list(set(image_files)))

        for img_path in image_files:
            total_images += 1
            
            # Ler imagem
            image = cv2.imread(img_path)
            if image is None:
                continue

            try:
                # Processar imagem
                hands_data = extractor.process_image_landmarks(image)
                
                # Se detectou alguma mão
                if hands_data:
                    for hand in hands_data:
                        # Achata os dados (63 coordenadas + hand side)
                        flat_data = extractor.flatten_hand_data(hand)
                        
                        # Adiciona a Label (Letra correspondente à pasta)
                        flat_data['label'] = label
                        
                        # (Opcional) Adicionar nome do arquivo para rastreabilidade
                        flat_data['source_image'] = os.path.basename(img_path)
                        
                        dataset_rows.append(flat_data)
                    processed_images += 1
            except Exception as e:
                print(f"Erro ao processar {img_path}: {e}")

    extractor.close()
    
    print(f"--- Processamento concluído ---")
    print(f"Total de imagens verificadas: {total_images}")
    print(f"Imagens com mãos detectadas: {processed_images}")
    
    if dataset_rows:
        df = pd.DataFrame(dataset_rows)
        
        # Reordenar colunas para que 'label' e 'hand' apareçam primeiro
        cols = ['label', 'hand'] + [c for c in df.columns if c not in ['label', 'hand', 'source_image']]
        df = df[cols]
        
        df.to_csv(output_file, index=False)
        print(f"Dataset salvo com sucesso em: {output_file}")
        print(f"Dimensões do dataset: {df.shape}")
    else:
        print("Nenhuma mão detectada. Dataset não criado.")


def run_minimal_camera_test() -> None:
    """
    Abre a câmera padrão, roda detecção de mãos, desenha keypoints e exibe.
    """
    print("--- Iniciando Teste de Câmera ---")
    extractor = HandLandmarkExtractor(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        suppress_warnings=True
    )

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera")
        extractor.close()
        return

    # Define resolução
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    prev_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)

            # Processar e desenhar landmarks
            hands_data = extractor.process_image_landmarks(frame)
            if hands_data:
                # pprint(hands_data) # Comentado para não poluir o terminal
                frame = extractor.draw_landmarks(frame, hands_data)
                cv2.putText(frame, f"Hands: {len(hands_data)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.putText(frame, "'q': sair", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(WINDOW_NAME, frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Selecione o modo de operação:")
    print("1. Gerar Dataset (Processar pasta SignAlphaSet)")
    print("2. Testar Câmera (Tempo real)")
    
    choice = input("Opção (1/2): ").strip()
    
    if choice == '1':
        generate_dataset_from_folder()
    elif choice == '2':
        run_minimal_camera_test()
    else:
        print("Opção inválida. A rodar geração de dataset por omissão...")
        generate_dataset_from_folder()