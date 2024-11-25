import tensorflow as tf
from tensorflow.keras import layers, Model

class my_model:

    def __init__(
            self,
            input_shape,
            base_model="EfficientNet",
            num_filters=64,
            class_num=1,
            mri_input_shape=(256,256,2),
            ct_input_shape=(256,256,5),):
        
        self.input_shape = input_shape
        self.input = layers.Input(self.input_shape)
        self.num_filters = num_filters
        self.class_num = class_num
        self.base_model = base_model
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

        x = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same", activation="relu")(input)
        x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
        x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
        x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
        x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
        return x

    def _architecture(self):

        encoder_mri = self._encoder(self.input[:,:,:,:2],"mri")
        encoder_ct = self._encoder(self.input[:,:,:,2:],"ct")

        bottleneck_features = self._bottleneck(encoder_mri, encoder_ct)
        x = self._decoder(bottleneck_features)

        self._head(x)

    def _head(self,input):
        x = layers.Conv2D(self.class_num, (1, 1))(input)
        if self.class_num == 1:
            self.output = layers.Activation('sigmoid')(x)
        else:
            self.output = layers.Activation('softmax')(x)


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