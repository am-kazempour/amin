import tensorflow as tf

def Dice_loss(y_true,y_pred):
    loss = 1 - (2 * tf.reduce_sum(y_true * y_pred)) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))
    return loss

def Exp_loss(y_true, y_pred):
    
    # y_true = tf.reshape(y_true, [-1])
    # y_pred = tf.reshape(y_pred, [-1])
    # y_pred = tf.nn.softmax(y_pred)
    true = tf.math.exp(-1*tf.math.multiply(y_true, y_pred))
    wrong = tf.math.exp(-1*tf.math.multiply((1-y_true), 1-y_pred))
    loss = tf.reduce_mean(true+wrong) 
    
    return loss