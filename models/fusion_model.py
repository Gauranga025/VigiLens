import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Concatenate
from tensorflow.keras.models import Model

def build_fusion_model():
    vis_input = Input(shape=(227, 227, 3))
    ir_input = Input(shape=(227, 227, 1))

    # Process visible
    vis = Conv2D(32, (3,3), activation='relu', padding='same')(vis_input)

    # Process IR
    ir = Conv2D(32, (3,3), activation='relu', padding='same')(ir_input)

    # Fuse
    fused = Concatenate()([vis, ir])
    fused = Conv2D(64, (3,3), activation='relu', padding='same')(fused)

    model = Model(inputs=[vis_input, ir_input], outputs=fused)
    return model