
from tensorflow.keras import backend as K
from tensorflow.python.keras.models import load_model

def pos_recall(y_true, y_pred):
    y_pred = K.cast(y_pred >= 0.9, 'float32')
    P = K.sum(y_true)
    TP = K.sum(y_pred * y_true)
    result = TP / P
    return result


def pos_precision(y_true, y_pred):
    # 预测为1
    y_pred = K.cast(y_pred >= 0.9, 'float32')
    P = K.sum(y_pred)
    TP = K.sum(y_pred * y_true)
    result = TP / (P+0.1)
    return result


model = load_model('./models/base_model_1/model.h5',
                   custom_objects={'pos_recall': pos_recall, 'pos_precision': pos_precision})
