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

    def conv_block(self,x, filters, kernel_size=3, activation='relu',repetition=2):
        for _ in range(repetition):
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
        
class SwinUNet(Unet):
    """
    Developed by Amin
    
    """
    def __init__(self,input_shape=(256,256, 3),num_filters=64,class_num=1,batch_norm=True,encoder_num=1,num_blocks=4,num_heads=4,head_dim=32,windows_size=4,shift_size=2):
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.head_dim =head_dim
        self.windows_size = windows_size
        self.shift_size = shift_size
        super().__init__(input_shape,num_filters,class_num,batch_norm,encoder_num)
    
    def _architecture(self):
        skip_connections = []
        x = self.input
        for _ in range(self.num_blocks):
            x = self._encoder(x)
            skip_connections.append(x)

        # Bottleneck
        x = self._block(x)

        # Decoder
        x = self._decoder(x, skip_connections)

        # Output layer (segmentation mask)
        x = layers.Conv2D(self.class_num, (1, 1))(x)
        if self.class_num == 1:
            self.output = layers.Activation('sigmoid')(x)
        else:
            self.output = layers.Activation('softmax')(x)

    
    def _encoder(self,inputs):
        x = inputs
        f = x.shape[-1] * 2
        for _ in range(self.num_blocks//2):
            x = self._block(x)
            x = layers.LayerNormalization()(x)
            x = layers.Conv2D(filters=f, kernel_size=3, padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        return x
    
    def _decoder(self, inputs, skip_connections):
        x = inputs
        for skip in reversed(skip_connections):
            x = layers.UpSampling2D((2,2))(x)
            x,skip = CustomPadding()(x,skip)
            x = layers.Concatenate()([x, skip])
            x = self._block(x)
            x = layers.LayerNormalization()(x)
        return x

    def _block(self, inputs):
        input_shape = inputs.shape
        height, width = input_shape[1], input_shape[2]

        # Partition the window into patches
        patch_size = self.windows_size * self.windows_size
        patches = ExtractPatchesLayer(self.windows_size)(inputs)

        # Multi-Head Attention with windows
        mha_layer = layers.MultiHeadAttention(key_dim=self.head_dim, num_heads=self.num_heads, attention_axes=(1, 2))
        attn_output = mha_layer(patches, patches)
        attn_output = layers.Reshape((height, width, -1))(attn_output)

        if attn_output.shape[-1] != inputs.shape[-1]:
            attn_output = layers.Conv2D(inputs.shape[-1], kernel_size=1, padding="same")(attn_output)
    
        # Shifted Window
        if self.shift_size > 0:
            attn_output = RollLayer(shift_size=self.shift_size, axis=1)(attn_output)

        # Skip connection
        return layers.Add()([inputs, attn_output])

class DeepLabv3(Unet):

    def __init__(self,input_shape=(256,256, 3),num_filters=64,class_num=1,batch_norm=True,encoder_num=1):
        super().__init__(input_shape,num_filters,class_num,batch_norm,encoder_num)
    
    def atrous_conv_block(self,x, filters, rate):
        x = layers.Conv2D(filters, kernel_size=3, padding='same', dilation_rate=rate)(x)
        x = layers.BatchNormalization()(x)
        return layers.ReLU()(x)
    
    def aspp_block(self,x):
        # ASPP with different atrous rates
        rate1 = 1
        rate2 = 6
        rate3 = 12
        rate4 = 18
        
        x1 = self.atrous_conv_block(x, 256, rate1)
        x2 = self.atrous_conv_block(x, 256, rate2)
        x3 = self.atrous_conv_block(x, 256, rate3)
        x4 = self.atrous_conv_block(x, 256, rate4)
        
        # Global Average Pooling
        x5 = layers.GlobalAveragePooling2D()(x)
        x5 = layers.Reshape((1, 1, -1))(x5)
        x5 = layers.Conv2D(256, kernel_size=1, padding='same')(x5)
        x5 = layers.BatchNormalization()(x5)
        x5 = layers.ReLU()(x5)
        x5 = layers.UpSampling2D(size=(tf.shape(x)[1], tf.shape(x)[2]), interpolation='bilinear')(x5)
        
        # Concatenate all features
        x = layers.Concatenate()([x1, x2, x3, x4, x5])
        # Final convolution
        x = self.conv_block(x,filters=256,kernel_size=1,repetition=1)
        return x

    # def decoder_block(self,x, skip):
    #     x = layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding='same')(x)
    #     x = layers.Concatenate()([x, skip])
    #     x = self.conv_block(x, 256)
    #     return x

    def _architecture(self):
        
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_tensor=self.input)
        layer_names = [
            "block3a_expand_activation",  
            "top_activation" 
        ]

        layers_outputs = [base_model.get_layer(name).output for name in layer_names]

        # ASPP block
        x = self.aspp_block(layers_outputs[1])

        # Decoder
        skip_connection = layers_outputs[0]
        skip_connection = self.conv_block(skip_connection,filters=48,kernel_size=1,repetition=1)
        
        x = layers.UpSampling2D(size=(tf.shape(skip_connection)[1] // tf.shape(x)[1], tf.shape(skip_connection)[2] // tf.shape(x)[2]), interpolation="bilinear")(x)
        x = layers.Concatenate()([x, skip_connection])
        
        x = self.conv_block(x,filters=256,kernel_size=3)
        # x = self.decoder_block(x, skip_connection)
        
        # Output layer
        x = layers.Conv2D(self.class_num, (1, 1))(x)
        if self.class_num == 1:
            self.output = layers.Activation('sigmoid')(x)
        else:
            self.output = layers.Activation('softmax')(x)
        
class CustomPadding(layers.Layer):
    def call(self, input1,input2):
        if input1.shape[1] > input2.shape[1]:
            input2 = tf.pad(input2, [[0, 0], [input1.shape[1] - input2.shape[1], 0], [0, 0], [0, 0]], "CONSTANT")
        elif input1.shape[1] < input2.shape[1]:
            input1 = tf.pad(input1, [[0, 0], [input2.shape[1] - input1.shape[1], 0], [0, 0], [0, 0]], "CONSTANT")
        if input1.shape[2] > input2.shape[2]:
            input2 = tf.pad(input2, [[0, 0], [0, 0], [input1.shape[2] - input2.shape[2], 0], [0, 0]], "CONSTANT")
        elif input1.shape[2] < input2.shape[2]:
            input1 = tf.pad(input1, [[0, 0], [0, 0], [input2.shape[2] - input1.shape[2], 0], [0, 0]], "CONSTANT")
        return input1,input2

class ExtractPatchesLayer(layers.Layer):
    def __init__(self, windows_size):
        super(ExtractPatchesLayer, self).__init__()
        self.windows_size = windows_size

    def call(self, inputs):
        patches = tf.image.extract_patches(inputs, 
                                           sizes=[1, self.windows_size, self.windows_size, 1],
                                           strides=[1, self.windows_size, self.windows_size, 1],
                                           rates=[1, 1, 1, 1], 
                                           padding='SAME')
        return patches
    
class RollLayer(layers.Layer):
    def __init__(self, shift_size, axis, **kwargs):
        super(RollLayer, self).__init__(**kwargs)
        self.shift_size = shift_size
        self.axis = axis

    def call(self, inputs):
        return tf.roll(inputs, shift=self.shift_size, axis=self.axis)
    
    def compute_output_shape(self, input_shape):
        return input_shape
