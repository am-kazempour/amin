import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
import tensorflow_wavelets.Layers.DWT as DWT

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

        self._head(x)

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

    def _head(self,input):
        x = layers.Conv2D(self.class_num, (1, 1))(input)
        if self.class_num == 1:
            self.output = layers.Activation('sigmoid')(x)
        else:
            self.output = layers.Activation('softmax')(x)

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
        
        self._head(c9)

class Unetpp(Unet):
    def _architecture(self):
        x40, x30, x20, x10, x00 = self._encoder(self.input)

        x40 = self.conv_block(x40,self.num_filters*16)

        x01 = self.conv_block(self.Concat(x00, x10), self.num_filters)
        x11 = self.conv_block(self.Concat(x10, x20), self.num_filters*2)
        x21 = self.conv_block(self.Concat(x20, x30), self.num_filters*4)
        x31 = self.conv_block(self.Concat(x30, x40), self.num_filters*8)
        
        x02 = self.conv_block(self.Concat(x00, x01, x11), self.num_filters)
        x12 = self.conv_block(self.Concat(x10, x11, x21), self.num_filters*2)
        x22 = self.conv_block(self.Concat(x20, x21, x31), self.num_filters*4)
        
        x03 = self.conv_block(self.Concat(x00, x01, x02, x12), self.num_filters)
        x13 = self.conv_block(self.Concat(x10, x11, x12, x22), self.num_filters*2)
        
        x04 = self.conv_block(self.Concat(x00, x01, x02, x03, x13), self.num_filters)

        self._head(x04)
    
    def Concat(self, input0, input1, input2=None, input3=None, input4=None):
        if input2 == None:
            return layers.Concatenate()([input0, layers.UpSampling2D(size=(2, 2))(input1)])
        elif input3 == None:
            return layers.Concatenate()([input0,input1, layers.UpSampling2D(size=(2, 2))(input2)])
        elif input4 == None:
            return layers.Concatenate()([input0,input1,input2, layers.UpSampling2D(size=(2, 2))(input3)])
        return layers.Concatenate()([input0,input1,input2,input3, layers.UpSampling2D(size=(2, 2))(input4)])
        
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
        
        x1 = self.atrous_conv_block(x, self.num_filters*4, rate1)
        x2 = self.atrous_conv_block(x, self.num_filters*4, rate2)
        x3 = self.atrous_conv_block(x, self.num_filters*4, rate3)
        x4 = self.atrous_conv_block(x, self.num_filters*4, rate4)
        
        # Global Average Pooling
        x5 = layers.GlobalAveragePooling2D()(x)
        x5 = layers.Reshape((1, 1, -1))(x5)
        x5 = layers.Conv2D(256, kernel_size=1, padding='same')(x5)
        x5 = layers.BatchNormalization()(x5)
        x5 = layers.ReLU()(x5)
        x5 = layers.UpSampling2D(size=(x.shape[1], x.shape[2]), interpolation='bilinear')(x5)
        
        # Concatenate all features
        x = layers.Concatenate()([x1, x2, x3, x4, x5])
        # Final convolution
        x = self.conv_block(x,filters=self.num_filters*4,kernel_size=1,repetition=1)
        return x

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
        skip_connection = self.conv_block(skip_connection,filters=self.num_filters,kernel_size=1,repetition=1)
        
        x = layers.UpSampling2D(size=(skip_connection.shape[1] // x.shape[1], skip_connection.shape[2] // x.shape[2]), interpolation="bilinear")(x)
        x = layers.Concatenate()([x, skip_connection])
        
        x = self.conv_block(x,filters=self.num_filters*4,kernel_size=3,repetition=1)
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)

        x = self.conv_block(x,filters=self.num_filters*4,kernel_size=3,repetition=1)
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        
        # Output layer
        x = layers.Conv2D(self.class_num, (1, 1))(x)
        if self.class_num == 1:
            self.output = layers.Activation('sigmoid')(x)
        else:
            self.output = layers.Activation('softmax')(x)

class Unet_skipBlock(Unet):

    def __init__(self, input_shape=(256, 256, 7), num_filters=64, class_num=1, batch_norm=True, encoder_num=1,dropout=0.25):
        self.depth1 = 32*2
        self.depth2 = 48*2
        self.depth3 = 64*2
        self.depth4 = 80*2
        self.depth5 = 128*2
        self.depth6 = 144*2
        self.depth7 = 160*2

        self.dropout = dropout

        self.encoder_skip1 = SkipBlock(filters = self.depth1,dropout=self.dropout,status="encoder")
        self.encoder_skip2 = SkipBlock(filters = self.depth3,dropout=self.dropout,status="encoder")
        self.encoder_skip3 = SkipBlock(filters = self.depth5,dropout=self.dropout,status="encoder")
        self.encoder_skip4 = SkipBlock(filters = self.depth7,dropout=self.dropout,status="encoder")

        self.decoder_skip1 = SkipBlock(filters = self.depth6,dropout=self.dropout,status="decoder")
        self.decoder_skip2 = SkipBlock(filters = self.depth4,dropout=self.dropout,status="decoder")
        self.decoder_skip3 = SkipBlock(filters = self.depth2,dropout=self.dropout,status="decoder")
        self.decoder_skip4 = SkipBlock(filters = self.depth1,dropout=self.dropout,status="decoder")

        super().__init__(input_shape, num_filters, class_num, batch_norm, encoder_num)
    
    def _encoder(self, input):
        x = Block([self.depth1,self.depth1],dropout=0.25)(input)
        skip = self.encoder_skip1(input)
        c1 = layers.Concatenate()([x,skip])
        print(c1.shape)
        x = Block([self.depth2,self.depth3],dropout=0.25)(c1)
        skip = self.encoder_skip2(c1)
        c2 = layers.Concatenate()([x,skip])
        print(c2.shape)
        x = Block([self.depth4,self.depth5],dropout=0.25)(c2)
        skip = self.encoder_skip3(c2)
        c3 = layers.Concatenate()([x,skip])
        print(c3.shape)
        x = Block([self.depth6,self.depth7],dropout=0.25)(c3)
        skip = self.encoder_skip4(c3)
        c4 = layers.Concatenate()([x,skip])
        print(c4.shape)
        return c4, c3, c2, c1

    def _decoder(self, c4, c3, c2, c1):
        x = Block([self.depth7,self.depth6],dropout=0.25,status="decoder")(c4)
        skip =  self.decoder_skip1(c4)
        x = layers.Concatenate()([x,skip])
        out = layers.Concatenate()([x,c3])
        print(out.shape)
        x = Block([self.depth5,self.depth4],dropout=0.25,status="decoder")(out)
        skip =  self.decoder_skip2(out)
        x = layers.Concatenate()([x,skip])
        out = layers.Concatenate()([x,c2])
        print(out.shape)
        x = Block([self.depth3,self.depth2],dropout=0.25,status="decoder")(out)
        skip =  self.decoder_skip3(out)
        x = layers.Concatenate()([x,skip])
        out = layers.Concatenate()([x,c1])
        print(out.shape)
        x = Block([self.depth1,self.depth1],dropout=0.25,status="decoder")(out)
        skip =  self.decoder_skip4(out)
        out = layers.Concatenate()([x,skip])
        print(out.shape)
        return out

    def _architecture(self):
        c4, c3, c2, c1 = self._encoder(self.input)
        x = self._decoder(c4, c3, c2, c1)
        self._head(x)
     
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

class Block(tf.keras.layers.Layer):
    def __init__(self, filters,status="encoder", kernel_size = (3, 3), he = 'he_normal', w = 4, strides = 1, padding = 'same', activation = 'relu',dropout=0.1):
        super(Block, self).__init__()
        self.status = status
        self.conv1 = layers.Conv2D(filters[0], kernel_size, strides=strides, padding=padding)#kernel_initializer=he, kernel_constraint=max_norm(w),
        self.conv2 = layers.Conv2D(filters[1], kernel_size, strides=strides, padding=padding)# kernel_initializer=he, kernel_constraint=max_norm(w),
        self.batch_norm1 = layers.BatchNormalization()
        self.batch_norm2 = layers.BatchNormalization()
        self.activation = layers.Activation(activation)
        self.maxPooling = layers.MaxPooling2D()
        self.upsample = layers.UpSampling2D(size=(2, 2))
        self.dropout = layers.Dropout(dropout)
        self.cbam = CBAM()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.activation(x)
        
        x = self.cbam(x)
        
        x = self.batch_norm1(x) 
        x = self.conv2(x)
        x = self.activation(x)
        x = self.batch_norm2(x)
        x = self.pad(x)
        if self.status == "encoder":
            x = self.maxPooling(x)
        elif self.status == "decoder":
            x = self.upsample(x)

        

        x = self.dropout(x)

        return x
    
    def pad(self,input):
        if input.shape[1] % 2 == 1:
            input = tf.pad(input,[[0,0],[1,0],[0,0],[0,0]], "CONSTANT")
        if input.shape[2] % 2 == 1:
             input= tf.pad(input,[[0,0],[0,0],[1,0],[0,0]], "CONSTANT")
        
        return input
        
class CBAM(layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        self.ratio = ratio
        super(CBAM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.channel_shared = layers.Dense(units=input_shape[-1] // self.ratio, activation='relu')
        self.channel_weight = layers.Dense(units=input_shape[-1], activation='sigmoid')
        self.spatial_weight = layers.Conv2D(filters=1, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='sigmoid')
        super(CBAM, self).build(input_shape)

    def call(self, inputs):
        # Channel-wise attention
        channel_avg = layers.GlobalAveragePooling2D()(inputs)
        channel_max = layers.GlobalMaxPooling2D()(inputs)
        channel_concat = layers.Add()([channel_avg, channel_max])
        channel_shared = self.channel_shared(channel_concat)
        channel_attention = self.channel_weight(channel_shared)
        channel_attention = layers.Multiply()([inputs, channel_attention])

        # Spatial attention
        spatial_avg = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        spatial_max = tf.reduce_max(inputs, axis=-1, keepdims=True)
        spatial_concat = tf.concat([spatial_avg, spatial_max], axis=-1)
        spatial_attention = self.spatial_weight(spatial_concat)
        spatial_attention = layers.Multiply()([inputs, spatial_attention])

        # Combine channel and spatial attention
        attention = layers.Add()([channel_attention, spatial_attention])

        return attention

class IWT(layers.Layer):
    def call(self, inputs):
        in_shape = tf.shape(inputs)
        batch_size = in_shape[0]
        height = in_shape[1]
        width = in_shape[2]
        # the number of channels can't be unknown for the convolutions
        n_channels = inputs.shape[3] // 4
        outputs = tf.zeros([batch_size, 2 * height, 2 * width, n_channels])
        # for now we only consider greyscale
        x1 = inputs[..., 0:n_channels] / 2
        x2 = inputs[..., n_channels:2*n_channels] / 2
        x3 = inputs[..., 2*n_channels:3*n_channels] / 2
        x4 = inputs[..., 3*n_channels:4*n_channels] / 2
        # in the following, E denotes even and O denotes odd
        x_EE = x1 - x2 - x3 + x4
        x_OE = x1 - x2 + x3 - x4
        x_EO = x1 + x2 - x3 - x4
        x_OO = x1 + x2 + x3 + x4

        # now the preparation to tensor_scatter_nd_add
        height_range_E = 2 * tf.range(height)
        height_range_O = height_range_E + 1
        width_range_E = 2 * tf.range(width)
        width_range_O = width_range_E + 1

        # this transpose allows to only index the varying dimensions
        # only the first dimensions can be indexed in tensor_scatter_nd_add
        # we also need to match the indices with the updates reshaping
        scatter_nd_perm = [2, 1, 3, 0]
        outputs_reshaped = tf.transpose(outputs, perm=scatter_nd_perm)

        combos_list = [
            ((height_range_E, width_range_E), x_EE),
            ((height_range_O, width_range_E), x_OE),
            ((height_range_E, width_range_O), x_EO),
            ((height_range_O, width_range_O), x_OO),
        ]
        for (height_range, width_range), x_comb in combos_list:
            h_range, w_range = tf.meshgrid(height_range, width_range)
            h_range = tf.reshape(h_range, (-1,))
            w_range = tf.reshape(w_range, (-1,))
            combo_indices = tf.stack([w_range, h_range], axis=-1)
            combo_reshaped = tf.transpose(x_comb, perm=scatter_nd_perm)
            outputs_reshaped =  tf.cond(
                batch_size > 0,
                lambda: tf.tensor_scatter_nd_add(
                    outputs_reshaped,
                    indices=combo_indices,
                    updates=tf.reshape(combo_reshaped, (-1, n_channels, batch_size)),
                ),
                lambda: outputs_reshaped,
            )

        inverse_scatter_nd_perm = [3, 1, 0, 2]
        outputs = tf.transpose(outputs_reshaped, perm=inverse_scatter_nd_perm)

        return outputs

class SkipBlock(layers.Layer):
    def __init__(self,filters,status="encoder", kernel_size=(1, 1),he = 'he_normal', w = 4, strides=(1, 1), padding='same', activation='relu',dropout=0.1):
        super(SkipBlock, self).__init__()
        self.dwt = DWT.DWT(concat=0)
        self.iwt = IWT()
        self.status = status
        self.cnn = layers.Conv2D(filters = filters, kernel_size=kernel_size, strides=strides, padding=padding)# kernel_initializer=he, kernel_constraint=max_norm(w),
        self.cnn2 = layers.Conv2D(filters = filters, kernel_size=kernel_size, strides=strides, padding=padding)# kernel_initializer=he, kernel_constraint=max_norm(w),
        self.batch_norm = layers.BatchNormalization()
        self.activation = layers.Activation(activation)
        self.dropout = layers.Dropout(dropout)
        self.max = layers.MaxPooling2D(2)
    
    def call(self,input):
        
        if self.status == "encoder":
            x = self.cnn(input) #64
            
            x = DWT.DWT(concat=0)(x)
           
        elif self.status == "decoder":
            x = self.cnn(input)
            x = self.iwt(x)
        
        x = self.cnn2(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x