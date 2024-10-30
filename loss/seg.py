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

def my_Exp_loss(y_true,y_pred):
    return tf.reduce_mean(tf.concat([tf.math.exp(y_pred[y_true==0]),tf.math.exp(-y_pred[y_true==1])],axis=0))

def my_Exp_w_loss(y_true,y_pred):
    y_p = tf.argmax(y_pred,axis=-1)
    y_p = tf.keras.utils.to_categorical(y_p,num_classes=y_pred.shape[-1])
    false_pos = 5*tf.math.exp(y_pred[y_p*(1-y_true) == 1])
    false_neg = 5*tf.math.exp(-y_pred[(1-y_p)*y_true == 1])
    l = tf.concat([tf.math.exp(y_pred[y_true==0]),tf.math.exp(-y_pred[y_true==1])],axis=0)
    return tf.reduce_mean(tf.concat([false_pos,false_neg,l],axis=0))