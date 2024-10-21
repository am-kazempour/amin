import tensorflow as tf

def Dice(y_true,y_pred):
    return (2 * tf.reduce_sum(y_true * y_pred)) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))

def IOU(y_true, y_pred):
    y_true = tf.expand_dims(y_true, axis=-1) if len(y_true.shape) == 3 else y_true
    y_pred = tf.expand_dims(y_pred, axis=-1) if len(y_pred.shape) == 3 else y_pred
    y_pred = tf.cast(y_pred >= 0.5, tf.float32)
    intersection = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
    union = tf.reduce_sum(tf.cast(y_true + y_pred, tf.float32)) - intersection
    iou = (intersection + 1e-10) / (union + 1e-10)
    return iou