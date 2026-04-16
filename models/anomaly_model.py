from tensorflow.keras.models import load_model

def load_anomaly_model():
    return load_model("model/best_model.keras")