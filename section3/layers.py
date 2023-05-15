# 모델 실행하는데 필요

# 라이브러리
import tensorflow as tf
from tensorflow.keras.layers import Layer

# L1 distance
class L1Dist(Layer):
    # 상속
    def __init__(self, **kwargs):
        super().__init__()
    # ?
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)