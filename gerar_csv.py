import os
import cv2
import pandas as pd
from hand_landmark_extractor import HandLandmarkExtractor

def create_asl_dataset(dataset_path: str, output_csv: str):
    """
    Itera sobre o dataset SignAlphaSet, extrai landmarks e guarda num CSV.
    """
    # Inicializa o extrator (modo estático para fotos)
    extractor = HandLandmarkExtractor(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        suppress_warnings=True
    )

    all_data_frames = []
    
    # 1. Itere sobre todas as pastas do dataset SignAlphaSet (A-Z)
    if not os.path.exists(dataset_path):
        print(f"Erro: A pasta {dataset_path} não foi encontrada.")
        return

    folders = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])

    for label in folders:
        folder_path = os.path.join(dataset_path, label)
        print(f"A processar letra: {label}...")

        # Itera sobre as imagens na pasta da letra
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            
            # Carrega a imagem
            image = cv2.imread(img_path)
            if image is None:
                continue

            # 2. Utilize a classe HandLandmarkExtractor para extrair os landmarks
            # Baseado na lógica do test_camera.py
            hands_data = extractor.process_image_landmarks(image)

            if hands_data:
                # 3. Cada landmark contém informação da mão e coordenadas (x, y, z)
                # O método hands_data_to_dataframe já aplana os 21 pontos em 63 colunas + identificação da mão
                df_temp = extractor.hands_data_to_dataframe(hands_data)
                
                # 4. A letra correspondente (label) obtém-se através da pasta
                df_temp['label'] = label
                
                all_data_frames.append(df_temp)

    # Resultado Esperado: Um novo dataset estruturado
    if all_data_frames:
        final_df = pd.concat(all_data_frames, ignore_index=True)
        final_df.to_csv(output_csv, index=False)
        print(f"Dataset criado com sucesso: {output_csv}")
        print(f"Total de registos: {len(final_df)}")
    else:
        print("Nenhum landmark foi detetado nas imagens.")

    extractor.close()

if __name__ == "__main__":
    DATASET_DIR = "SignAlphaSet"
    OUTPUT_FILE = "asl_landmarks_dataset.csv"
    create_asl_dataset(DATASET_DIR, OUTPUT_FILE)