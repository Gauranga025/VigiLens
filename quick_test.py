import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/saved_model.keras")

# Dummy input (same shape as training)
dummy_input = np.random.rand(1, 227, 227, 10, 1)

output = model.predict(dummy_input)

print("✅ Prediction successful")
print("Output shape:", output.shape)