import tensorflow as tf

def Dice_loss(y_true,y_pred):
    loss = 1 - (2 * tf.reduce_sum(y_true * y_pred)) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))
    return loss