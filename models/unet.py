import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
class Unet:
    """
    Developed by Amin
    
    """
    def __init__(self,input_shape=(256,256, 3),num_filters=64,class_num=1,batch_norm=True,encoder_num=1):
        self.input_shape = input_shape
        self.input = layers.Input(self.input_shape)
        self.num_filters = num_filters
        self.class_num = class_num
        self.batch_norm = batch_norm
        self.encoder_num = encoder_num
        self._architecture()

    def model(self):
        return Model(inputs = self.input,outputs = [self.output])

    def _architecture(self):
        if self.encoder_num == 1:
            output, c4, c3, c2, c1= self._encoder(self.input)
        else:
            output1, c41, c31, c21, c11= self._encoder(self.input[:,:,:,:2])
            output2, c42, c32, c22, c12= self._encoder(self.input[:,:,:,2:])
            output = layers.concatenate([output1,output2])
            c4 = layers.concatenate([c41,c42])
            c3 = layers.concatenate([c31,c32])
            c2 = layers.concatenate([c21,c22])
            c1 = layers.concatenate([c11,c12])
        
        output = self._bottleneck(output)
        x = self._decoder(output, c4, c3, c2, c1)

        x = layers.Conv2D(self.class_num, (1, 1))(x)
        if self.class_num == 1:
            self.output = layers.Activation('sigmoid')(x)
        else:
            self.output = layers.Activation('softmax')(x)

    def _encoder(self,input):

        c1 = self.conv_block(input,self.num_filters)
        x = layers.MaxPooling2D((2, 2))(c1)

        c2 = self.conv_block(x, self.num_filters*2)
        x = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = self.conv_block(x, self.num_filters*4)
        x = layers.MaxPooling2D((2, 2))(c3)
        
        c4 = self.conv_block(x, self.num_filters*8)
        x = layers.MaxPooling2D((2, 2))(c4)
        
        return x, c4, c3, c2, c1

    def _bottleneck(self,input,activation='relu'):
        x = layers.Conv2D(self.num_filters*16, (3, 3), padding='same')(input)
        if self.batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Conv2D(self.num_filters*16, (3, 3),  padding='same')(x)
        if self.batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        return x

    def _decoder(self, input, c4, c3, c2, c1):
        
        x = layers.UpSampling2D((2, 2))(input)
        x = layers.concatenate([x, c4])
        x = self.conv_block(x,self.num_filters*8)

        x = layers.UpSampling2D((2, 2))(x)
        x = layers.concatenate([x, c3])
        x = self.conv_block(x,self.num_filters*4)

        x = layers.UpSampling2D((2, 2))(x)
        x = layers.concatenate([x, c2])
        x = self.conv_block(x,self.num_filters*2)

        x = layers.UpSampling2D((2, 2))(x)
        x = layers.concatenate([x, c1])
        x = self.conv_block(x,self.num_filters)

        return x

    def conv_block(self,x, filters, kernel_size=3, activation='relu'):
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        if self.batch_norm:
            x = layers.BatchNormalization()(x)
        
        x = layers.Activation(activation)(x)
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        
        if self.batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        return x


class trans_unet(Unet):
    """
    Developed by Amin
    
    """
    def __init__(self,input_shape=(256,256, 3),num_filters=64,class_num=1,num_heads=4,ff_dim=32,batch_norm=True,encoder_num=1):
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        super().__init__(input_shape,num_filters,class_num,batch_norm,encoder_num)

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
 
    def _architecture(self):
        
        if self.encoder_num == 1:
            output, c4, c3, c2, c1= self._encoder(self.input)
        else:
            output1, c41, c31, c21, c11= self._encoder(self.input[:,:,:,:2])
            output2, c42, c32, c22, c12= self._encoder(self.input[:,:,:,2:])
            output = layers.concatenate([output1,output2])
            c4 = layers.concatenate([c41,c42])
            c3 = layers.concatenate([c31,c32])
            c2 = layers.concatenate([c21,c22])
            c1 = layers.concatenate([c11,c12])

        # c5 = tf.reshape(c5,(256,2024))
        # Transformer Encoder
        shape_before_flattening = K.int_shape(output)
        x = layers.Reshape((shape_before_flattening[1] * shape_before_flattening[2], self.num_filters*8*self.encoder_num))(output)
        x = self.transformer_encoder(x)
        x = layers.Reshape((shape_before_flattening[1], shape_before_flattening[2], self.num_filters*8*self.encoder_num))(x)
        
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
        
class SwinUNet:
    pass
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