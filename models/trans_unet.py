import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import keras

class trans_unet:
    """
    Developed by Amin
    
    """
    def __init__(self,input_shape=(256,256, 3),num_filters=64,class_num=1,num_heads=4,ff_dim=32):
        self.input_shape = input_shape
        self.input = Input(self.input_shape)
        self.num_filters = num_filters
        self.class_num = class_num
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.__architecture()

    def model(self):
      return keras.Model(inputs = self.input,outputs = [self.output])

    def transformer_encoder(self , x):
        # Normalization and Attention
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim= self.ff_dim)(x1, x1)
        x2 = layers.Add()([attention_output, x])
        
        # Feed Forward Network
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(self.ff_dim, activation='relu')(x3)
        x3 = layers.Dense(x.shape[-1])(x3)
        x_out = layers.Add()([x3, x2])
        return x_out

    def __architecture(self):
        
        # Encoder: UNet with Conv Blocks
        c1 = self.conv_block(self.input, self.num_filters)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = self.conv_block(p1, self.num_filters*2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = self.conv_block(p2, self.num_filters*4)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        
        c4 = self.conv_block(p3, self.num_filters*8)
        p4 = layers.MaxPooling2D((2, 2))(c4)
        
        c5 = self.conv_block(p4, self.num_filters*16)
        
        # Transformer Encoder
        shape_before_flattening = K.int_shape(c5)
        x = layers.Reshape((shape_before_flattening[1] * shape_before_flattening[2], self.num_filters*16))(c5)
        x = self.transformer_encoder(x)
        x = layers.Reshape((shape_before_flattening[1], shape_before_flattening[2], self.num_filters*16))(x)
        
        # Decoder: UNet with Conv Blocks
        u6 = layers.UpSampling2D((2, 2))(x)
        u6,c4 = CustomPadding()(u6,c4)
        u6 = layers.Concatenate()([u6, c4])
        c6 = self.conv_block(u6, self.num_filters*8)
        
        u7 = layers.UpSampling2D((2, 2))(c6)
        u7,c3 = CustomPadding()(u7,c3)
        u7 = layers.Concatenate()([u7, c3])
        c7 = self.conv_block(u7, self.num_filters*4)
        
        u8 = layers.UpSampling2D((2, 2))(c7)
        u8,c2 = CustomPadding()(u8,c2)
        u8 = layers.Concatenate()([u8, c2])
        c8 = self.conv_block(u8, self.num_filters*2)
        
        u9 = layers.UpSampling2D((2, 2))(c8)
        u9,c1 = CustomPadding()(u9,c1)
        u9 = layers.Concatenate()([u9, c1])
        c9 = self.conv_block(u9, self.num_filters)
        
        c9 = layers.Conv2D(self.class_num, (1, 1))(c9)
        if self.class_num == 1:
            self.output = layers.Activation('sigmoid')(c9)
        else:
            self.output = layers.Activation('softmax')(c9)
        
    def conv_block(self,x, filters, kernel_size=3, activation='relu'):
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        return x


class trans_unet_1:
    """
    Developed by Amin
    
    """
    def __init__(self,input_shape=(256,256, 3),num_filters=64,class_num=1,num_heads=4,ff_dim=32):
        self.input_shape = input_shape
        self.input = Input(self.input_shape)
        self.num_filters = num_filters
        self.class_num = class_num
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.__architecture()

    def model(self):
      return keras.Model(inputs = self.input,outputs = [self.output])

    def transformer_encoder(self , x):
        # Normalization and Attention
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim= self.ff_dim)(x1, x1)
        x2 = layers.Add()([attention_output, x])
        
        # Feed Forward Network
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(self.ff_dim, activation='relu')(x3)
        x3 = layers.Dense(x.shape[-1])(x3)
        x_out = layers.Add()([x3, x2])
        return x_out

    def _encoder(self,input):
        
        # Encoder: UNet with Conv Blocks
        c1 = self.conv_block(input, self.num_filters)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = self.conv_block(p1, self.num_filters*2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = self.conv_block(p2, self.num_filters*4)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        
        c4 = self.conv_block(p3, self.num_filters*8)
        p4 = layers.MaxPooling2D((2, 2))(c4)
        
        c5 = self.conv_block(p4, self.num_filters*16)

        return c5
        
    def __architecture(self):
        
        e1 = self._encoder(self.input[:,:,:,:2])
        e2 = self._encoder(self.input[:,:,:,2:])
        
        c5 = layers.Concatenate()([e1, e2])

        # Transformer Encoder
        shape_before_flattening = K.int_shape(c5)
        x = layers.Reshape((shape_before_flattening[1] * shape_before_flattening[2], self.num_filters*16))(c5)
        x = self.transformer_encoder(x)
        x = layers.Reshape((shape_before_flattening[1], shape_before_flattening[2], self.num_filters*16))(x)
        
        # Decoder: UNet with Conv Blocks
        u6 = layers.UpSampling2D((2, 2))(x)
        u6,c4 = CustomPadding()(u6,c4)
        u6 = layers.Concatenate()([u6, c4])
        c6 = self.conv_block(u6, self.num_filters*8)
        
        u7 = layers.UpSampling2D((2, 2))(c6)
        u7,c3 = CustomPadding()(u7,c3)
        u7 = layers.Concatenate()([u7, c3])
        c7 = self.conv_block(u7, self.num_filters*4)
        
        u8 = layers.UpSampling2D((2, 2))(c7)
        u8,c2 = CustomPadding()(u8,c2)
        u8 = layers.Concatenate()([u8, c2])
        c8 = self.conv_block(u8, self.num_filters*2)
        
        u9 = layers.UpSampling2D((2, 2))(c8)
        u9,c1 = CustomPadding()(u9,c1)
        u9 = layers.Concatenate()([u9, c1])
        c9 = self.conv_block(u9, self.num_filters)
        
        c9 = layers.Conv2D(self.class_num, (1, 1))(c9)
        if self.class_num == 1:
            self.output = layers.Activation('sigmoid')(c9)
        else:
            self.output = layers.Activation('softmax')(c9)
        
    def conv_block(self,x, filters, kernel_size=3, activation='relu'):
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        return x

class CustomPadding(layers.Layer):
    def call(self, input1,input2):
        if input1.shape[1] > input2.shape[1]:
            input2 = tf.pad(input2, [[0, 0], [1, 0], [0, 0], [0, 0]], "CONSTANT")
        elif input1.shape[1] < input2.shape[1]:
            input1 = tf.pad(input1, [[0, 0], [1, 0], [0, 0], [0, 0]], "CONSTANT")
        if input1.shape[2] > input2.shape[2]:
            input2 = tf.pad(input2, [[0, 0], [0, 0], [1, 0], [0, 0]], "CONSTANT")
        elif input1.shape[2] < input2.shape[2]:
            input1 = tf.pad(input1, [[0, 0], [0, 0], [1, 0], [0, 0]], "CONSTANT")
        return input1,input2