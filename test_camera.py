"""
Teste mínimo de câmera: envia frames ao HandLandmarkExtractor e exibe landmarks.

Teclas:
- 'q': sair
- 's': salvar CSV com landmarks detectados e imagem atual com overlay
"""

import cv2
from hand_landmark_extractor import HandLandmarkExtractor
from typing import Optional
import os
import time
from pprint import pprint

# Constantes de janela e gravação
WINDOW_NAME = "Hand Landmarks"
CSV_OUTPUT_PATH = "1_landmarks.csv"
IMG_OUTPUT_PATH = "1_landmarks.jpg"
CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def run_minimal_camera_test() -> None:
    """
    Abre a câmera padrão, roda detecção de mãos, desenha keypoints e exibe.
    """
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

            # Espelhar para experiência mais natural
            frame = cv2.flip(frame, 1)

            # Processar e desenhar landmarks
            hands_data = extractor.process_image_landmarks(frame)
            if hands_data:
                pprint(hands_data)
                frame = extractor.draw_landmarks(frame, hands_data)
                cv2.putText(frame, f"Hands: {len(hands_data)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # FPS (opcional)
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Overlay de ajuda
            cv2.putText(frame, "q: sair  s: salvar CSV+IMG", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and hands_data:
                # Salva CSV (append) e imagem atual
                df = extractor.hands_data_to_dataframe(hands_data)
                try:
                    file_exists = os.path.exists(CSV_OUTPUT_PATH)
                    df.to_csv(CSV_OUTPUT_PATH, mode='a', header=not file_exists, index=False)
                    cv2.imwrite(IMG_OUTPUT_PATH, frame)
                    print(f"Saved: {CSV_OUTPUT_PATH}, {IMG_OUTPUT_PATH}")
                except Exception as e:
                    print(f"Error saving to {CSV_OUTPUT_PATH} / {IMG_OUTPUT_PATH}: {e}")
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_minimal_camera_test()
