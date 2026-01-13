import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Carregar o modelo treinado
with open('melhor_modelo.pkl', 'rb') as f:
    checkpoint = pickle.load(f)
    model = checkpoint['model']
    inv_label_map = checkpoint['inv_label_map']

def pre_processamento(data_dict):
    df = pd.DataFrame([data_dict])
    
    x_cols = [c for c in df.columns if c.endswith("_x")]
    y_cols = [c for c in df.columns if c.endswith("_y")]
    z_cols = [c for c in df.columns if c.endswith("_z")]
    lm_cols = x_cols + y_cols + z_cols

    # 1. Espelhamento
    if 'hand' in df.columns:
        is_left = df["hand"].astype(str).str.lower() == "left"
        if is_left.any():
            df.loc[is_left, x_cols] = 1.0 - df.loc[is_left, x_cols]
        df["is_right"] = (~is_left).astype(int)
        df = df.drop(columns=["hand"])

    # 2. Centralização no Pulso
    df[x_cols] = df[x_cols].sub(df["WRIST_x"], axis=0)
    df[y_cols] = df[y_cols].sub(df["WRIST_y"], axis=0)
    df[z_cols] = df[z_cols].sub(df["WRIST_z"], axis=0)

    # 3. Scaling
    scale = np.sqrt(df["MIDDLE_FINGER_MCP_x"]**2 + df["MIDDLE_FINGER_MCP_y"]**2 + df["MIDDLE_FINGER_MCP_z"]**2)
    df[lm_cols] = df[lm_cols].div(scale.replace(0, 1), axis=0)

    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        X_processed = pre_processamento(data)
        
        prediction_idx = model.predict(X_processed)[0]
        letter = inv_label_map[prediction_idx]
        
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            confidence = float(np.max(model.predict_proba(X_processed)[0]))
            
        return jsonify({'letter': str(letter), 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)