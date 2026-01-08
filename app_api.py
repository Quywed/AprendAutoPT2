from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# 1. Carregar o modelo e os transformadores
MODEL_PATH = 'melhor_modelo.pkl'
ENCODER_PATH = 'hand_encoder.pkl'
SCALER_PATH = 'scaler.pkl'

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(ENCODER_PATH, 'rb') as f:
    le_hand = pickle.load(f)

# Carregar o scaler guardado durante o treino
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Sem dados'}), 400
            
        # Cria DataFrame a partir do dicionário recebido
        # É importante que o dicionário contenha as 64 colunas (hand + 63 landmarks)
        X = pd.DataFrame([data])
        
        # 2. Codificar a coluna 'hand' (ex: "Left" -> 0)
        if 'hand' in X.columns:
            X['hand'] = le_hand.transform(X['hand'])
        
        # 3. Aplicar o Scaler
        # O transform deve ser aplicado a todas as colunas na mesma ordem do treino
        X_scaled = scaler.transform(X)
        
        # 4. Fazer a previsão usando os dados escalonados
        prediction = model.predict(X_scaled)[0]
        
        # Obter probabilidades para a confiança
        probabilities = model.predict_proba(X_scaled)[0]
        confidence = float(np.max(probabilities))
        
        return jsonify({
            'letter': str(prediction),
            'confidence': confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok', 
        'model_loaded': True,
        'scaler_loaded': True
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)