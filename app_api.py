from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# 1. Carregar o modelo, o Label Encoder e (se necessário) o Scaler
# Se o seu 'melhor_modelo.pkl' for um Pipeline do sklearn, o scaler já está lá dentro.
MODEL_PATH = 'melhor_modelo.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(LABEL_ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

def preprocess_for_inference(data_dict):
    """
    Replica o pré-processamento do notebook para um único exemplo.
    """
    df = pd.DataFrame([data_dict])
    
    # Identificar colunas
    x_cols = [c for c in df.columns if c.endswith("_x")]
    y_cols = [c for c in df.columns if c.endswith("_y")]
    z_cols = [c for c in df.columns if c.endswith("_z")]
    lm_cols = x_cols + y_cols + z_cols

    # 1. Mirroring (Espelhar mão esquerda para referencial da direita)
    if 'hand' in df.columns:
        is_left = df["hand"].astype(str).str.lower() == "left"
        if is_left.any():
            df.loc[is_left, x_cols] = 1.0 - df.loc[is_left, x_cols]
        
        # Criar a feature binária que o modelo espera
        df["is_right"] = (~is_left).astype(int)
        df = df.drop(columns=["hand"])

    # 2. Normalização Geométrica (Centralizar no WRIST e escalar)
    # Subtrair coordenadas do Pulso (WRIST)
    df[x_cols] = df[x_cols].sub(df["WRIST_x"], axis=0)
    df[y_cols] = df[y_cols].sub(df["WRIST_y"], axis=0)
    df[z_cols] = df[z_cols].sub(df["WRIST_z"], axis=0)

    # Calcular escala (distância WRIST -> MIDDLE_FINGER_MCP)
    scale = np.sqrt(
        df["MIDDLE_FINGER_MCP_x"]**2 + 
        df["MIDDLE_FINGER_MCP_y"]**2 + 
        df["MIDDLE_FINGER_MCP_z"]**2
    )
    df[lm_cols] = df[lm_cols].div(scale.replace(0, 1), axis=0)

    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Sem dados'}), 400
            
        # 1. Pré-processamento
        X_processed = preprocess_for_inference(data)
        
        # 2. Previsão
        # Se o modelo for um Pipeline, ele aplica o StandardScaler automaticamente
        prediction_encoded = model.predict(X_processed)[0]
        
        # 3. Descodificar a letra (de número para 'A', 'B', etc.)
        letter = le.inverse_transform([prediction_encoded])[0]
        
        # 4. Probabilidades para confiança
        probabilities = model.predict_proba(X_processed)[0]
        confidence = float(np.max(probabilities))
        
        # 5. Resposta incluindo landmarks (conforme o to-do.txt)
        return jsonify({
            'letter': str(letter),
            'confidence': confidence,
            'landmarks_received': data  # Retorna os dados originais para conferência
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok', 
        'model_loaded': model is not None,
        'label_encoder_loaded': le is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)