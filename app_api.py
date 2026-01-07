from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Carregar o modelo treinado
MODEL_PATH = 'melhor_modelo.pkl'
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Adicione o carregamento do encoder guardado no treino
# Adicione o carregamento do encoder guardado no treino
with open('hand_encoder.pkl', 'rb') as f:
    le_hand = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Sem dados'}), 400
            
        # Cria DataFrame a partir do dicionário recebido (mantém os nomes das colunas)
        X = pd.DataFrame([data])
        
        # 1. Codificar a coluna 'hand' (ex: "Left" -> 0)
        if 'hand' in X.columns:
            X['hand'] = le_hand.transform(X['hand'])
        
        # 2. Fazer a previsão (X já tem as 64 colunas com nomes corretos)
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        confidence = float(np.max(probabilities))
        
        return jsonify({
            'letter': str(prediction),
            'confidence': confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
