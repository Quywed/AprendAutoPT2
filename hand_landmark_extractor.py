"""
Hand Landmark Extractor for MediaPipe

This module provides a class to extract and process hand landmark coordinates
from MediaPipe hand detection results.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
import warnings
import pandas as pd
from typing import Dict as _DictAlias

# Tipos auxiliares
HandData = _DictAlias[str, Any]
HandsDataList = List[HandData]
LandmarksArray = np.ndarray  # (21, 3) normalizado ou (21, 2) em pixels

# Conexões padrão entre marcos da mão
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),  # Indicador
    (0, 9), (9, 10), (10, 11), (11, 12),  # Médio
    (0, 13), (13, 14), (14, 15), (15, 16),  # Anelar
    (0, 17), (17, 18), (18, 19), (19, 20),  # Mínimo
    (5, 9), (9, 13), (13, 17)  # Palma
]


class HandLandmarkExtractor:
    """
    Extrai e processa coordenadas de marcos (landmarks) de mãos via MediaPipe.
    """
    
    def __init__(self, 
                 static_image_mode: bool = True,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 suppress_warnings: bool = True):
        
        if suppress_warnings:
            logging.getLogger('mediapipe').setLevel(logging.ERROR)
            warnings.filterwarnings('ignore', category=UserWarning)
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        except Exception as e:
            print(f"Warning: MediaPipe initialization issue: {e}")
            # Re-initialize with default settings if fail
            self.hands = self.mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        
        self.landmark_names = [
            'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
            'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
            'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
            'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
            'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
        ]

    def process_image_landmarks(self, image: np.ndarray) -> List[HandData]:
        """
        Processa uma imagem e retorna APENAS landmarks normalizados por mão detectada.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image deve ser um np.ndarray (BGR)")
        if image.ndim not in (2, 3):
            raise ValueError("image deve ter 2 ou 3 dimensões (HxW ou HxWxC)")
        
        # Converte BGR para RGB para o MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        output: List[Dict[str, Any]] = []
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                output.append({
                    'handedness': self._get_handedness(results.multi_handedness, idx),
                    'landmarks': self._extract_landmarks(hand_landmarks),
                    'landmarks_normalized': self._extract_landmarks_normalized(hand_landmarks)
                })
        return output

    def process_image_landmarks_pixel(self, image: np.ndarray) -> List[HandData]:
        if not isinstance(image, np.ndarray):
            raise TypeError("image deve ser um np.ndarray (BGR)")
        if image.ndim not in (2, 3):
            raise ValueError("image deve ter 2 ou 3 dimensões (HxW ou HxWxC)")
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        output: List[Dict[str, Any]] = []
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                output.append({
                    'handedness': self._get_handedness(results.multi_handedness, idx),
                    'landmarks_pixel': self._extract_landmarks_pixel(hand_landmarks, image.shape)
                })
        return output

    def flatten_hand_data(self, hand_data: HandData) -> Dict[str, Any]:
        """
        Converte os dados de UMA mão num dicionário plano (flat) com colunas x, y, z.
        Útil para criar datasets.
        """
        handedness = hand_data.get('handedness', 'Unknown')
        ln = hand_data.get('landmarks_normalized', None)
        landmarks = hand_data.get('landmarks', [])

        flat: Dict[str, Any] = { 'hand': handedness }

        # Prefer normalized ndarray if available
        if isinstance(ln, np.ndarray) and ln.ndim == 2 and ln.shape[0] >= 1 and ln.shape[1] >= 3:
            count = min(len(self.landmark_names), ln.shape[0])
            for idx in range(count):
                name = self.landmark_names[idx]
                flat[f'{name}_x'] = float(ln[idx, 0])
                flat[f'{name}_y'] = float(ln[idx, 1])
                flat[f'{name}_z'] = float(ln[idx, 2])
        else:
            # Fallback
            count = min(len(self.landmark_names), len(landmarks))
            for idx in range(count):
                name = self.landmark_names[idx]
                lm = landmarks[idx]
                flat[f'{name}_x'] = float(lm.get('x', 0.0))
                flat[f'{name}_y'] = float(lm.get('y', 0.0))
                flat[f'{name}_z'] = float(lm.get('z', 0.0))
        
        return flat

    def hands_data_to_dataframe(self, hands_data: HandsDataList) -> 'pd.DataFrame':
        """
        Converte uma lista de dicionários de mãos em DataFrame.
        """
        rows = [self.flatten_hand_data(hd) for hd in hands_data]
        return pd.DataFrame(rows)

    def process_image_to_dataframe(self, image: np.ndarray) -> 'pd.DataFrame':
        hands = self.process_image_landmarks(image)
        return self.hands_data_to_dataframe(hands)

    def _extract_landmarks(self, hand_landmarks) -> List[Dict[str, float]]:
        landmarks = []
        for idx, landmark in enumerate(hand_landmarks.landmark):
            landmark_data = {
                'name': self.landmark_names[idx],
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': getattr(landmark, 'visibility', 1.0)
            }
            landmarks.append(landmark_data)
        return landmarks
    
    def _extract_landmarks_normalized(self, hand_landmarks) -> LandmarksArray:
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    
    def _extract_landmarks_pixel(self, hand_landmarks, image_shape: Tuple[int, int, int]) -> LandmarksArray:
        height, width = image_shape[:2]
        landmarks = []
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks.append([x, y])
        return np.array(landmarks)
    
    def _get_handedness(self, multi_handedness, hand_idx: int) -> str:
        if multi_handedness and hand_idx < len(multi_handedness):
            return multi_handedness[hand_idx].classification[0].label
        return 'Unknown'
    
    def draw_landmarks(self, image: np.ndarray, landmarks_data: List[HandData]) -> np.ndarray:
        output_image = image.copy()
        height, width = image.shape[:2]
        
        for hand_data in landmarks_data:
            landmarks = hand_data['landmarks']
            for landmark in landmarks:
                x = int(landmark['x'] * width)
                y = int(landmark['y'] * height)
                cv2.circle(output_image, (x, y), 3, (0, 255, 0), -1)
            
            for start_idx, end_idx in HAND_CONNECTIONS:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_x = int(landmarks[start_idx]['x'] * width)
                    start_y = int(landmarks[start_idx]['y'] * height)
                    end_x = int(landmarks[end_idx]['x'] * width)
                    end_y = int(landmarks[end_idx]['y'] * height)
                    cv2.line(output_image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
        return output_image
    
    def close(self) -> None:
        if hasattr(self, 'hands') and self.hands is not None:
            try:
                self.hands.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass