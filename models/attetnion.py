import tensorflow as tf
from tensorflow.keras import layers, Model

from .unet import Unet

class my_model:

    def __init__(
            self,
            input_shape,
            base_model="EfficientNet",
            num_filters=64,
            class_num=1,
            batch_norm=True,
            mri_input_shape=(256,256,2),
            ct_input_shape=(256,256,5),):
        
        self.input_shape = input_shape
        self.input = layers.Input(self.input_shape)
        self.num_filters = num_filters
        self.class_num = class_num
        self.base_model = base_model
        self.batch_norm = batch_norm
        self.mri_input_shape = mri_input_shape
        self.ct_input_shape = ct_input_shape
        self._architecture()

    def model(self):
        return Model(inputs = self.input,outputs = [self.output])
    
    def _encoder(self,input,name):

        if self.base_model == "EfficientNet":
            encoder = tf.keras.applications.EfficientNetB0(
                include_top=False,
                input_tensor=input,
                weights=None)
        for layer in encoder.layers:
            layer.name = f"{name}_{layer.name}"
        
        x = encoder.output
        x = layers.Conv2D(512, (1, 1), activation="relu")(x)
        return x

    def _cross_attention(self,query, key, value):

        attended_values = CrossAttentionLayer(units=512)([query, key, value])
        return attended_values

    def _bottleneck(self,encoder_mri, encoder_ct):

        mri_to_ct = self._cross_attention(encoder_mri, encoder_ct, encoder_ct)

        ct_to_mri = self._cross_attention(encoder_ct, encoder_mri, encoder_mri)

        combined = layers.Concatenate()([mri_to_ct, ct_to_mri])
        combined = layers.Dense(512, activation="relu")(combined)
        combined = layers.Reshape((8, 8, 512))(combined)  # Reshape back to spatial format
        return combined

    def _decoder(self,input):
        x = input
        fillters = 512
        for _ in range(5):
            x = layers.UpSampling2D(size=(2, 2))(x)
            x = self.conv_block(x,filters=fillters)
            fillters //= 2


        # x = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same", activation="relu")(input)
        # x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
        # x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
        # x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
        # x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
        return x

    def _architecture(self):

        encoder_mri = self._encoder(self.input[:,:,:,:2],"mri")
        encoder_ct = self._encoder(self.input[:,:,:,2:],"ct")

        bottleneck_features = self._bottleneck(encoder_mri, encoder_ct)
        x = self._decoder(bottleneck_features)

        self._head(x)

    def conv_block(self,x, filters, kernel_size=3, activation='relu',repetition=1):
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

class my_Unet(Unet):

    def __init__(
            self,
            input_shape=(256,256,7),
            num_filters=64,
            class_num=1,
            batch_norm=True,
            encoder_num=1,
            mri_input_shape=(256,256,2),
            ct_input_shape=(256,256,5),):
        
        self.mri_input_shape = mri_input_shape
        self.ct_input_shape = ct_input_shape
        super().__init__(input_shape, num_filters, class_num, batch_norm, encoder_num)
    
    def _cross_attention(self,query, key, value):

        attended_values = CrossAttentionLayer(units=512)([query, key, value])
        return attended_values

    def _bottleneck(self,encoder_mri, encoder_ct):

        mri_to_ct = self._cross_attention(encoder_mri, encoder_ct, encoder_ct)

        ct_to_mri = self._cross_attention(encoder_ct, encoder_mri, encoder_mri)

        combined = layers.Concatenate()([mri_to_ct, ct_to_mri])
        combined = layers.Dense(512, activation="relu")(combined)
        combined = layers.Reshape((16, 16, 512))(combined)  # Reshape back to spatial format
        return combined

    def _architecture(self):

        output1, c41, c31, c21, c11= self._encoder(self.input[:,:,:,:2])
        output2, c42, c32, c22, c12= self._encoder(self.input[:,:,:,2:])

        c4 = layers.concatenate([c41,c42])
        c3 = layers.concatenate([c31,c32])
        c2 = layers.concatenate([c21,c22])
        c1 = layers.concatenate([c11,c12])

        bottleneck_features = self._bottleneck(output1, output2)
        x = self._decoder(bottleneck_features, c4, c3, c2, c1)

        self._head(x)

class CrossAttentionLayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super(CrossAttentionLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.query_dense = layers.Dense(self.units)
        self.key_dense = layers.Dense(self.units)
        self.value_dense = layers.Dense(self.units)
        super(CrossAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        query, key, value = inputs
        # محاسبه query, key, value
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        
        # محاسبه attention scores
        scores = tf.matmul(query, key, transpose_b=True)
        scores = tf.nn.softmax(scores, axis=-1)
        
        # اعمال توجه و گرفتن مقادیر
        attended_values = tf.matmul(scores, value)
        return attended_values