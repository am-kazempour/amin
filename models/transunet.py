import tensorflow as tf
import tensorflow_addons as tfa
import math

L2_WEIGHT_DECAY = 1e-4

class TransUnet:
    def __init__(
                self,
                input_shape=(256,256, 3), 
                patch_size=16, 
                hybrid=True,
                grid=(14,14), 
                hidden_size=768,
                n_layers=12,
                n_heads=12,
                mlp_dim=3072,
                dropout=0.1,
                decoder_channels=[256,128,64,16],
                n_skip=3,
                num_classes=3,
                final_act='sigmoid',
                pretrain=True,
                freeze_enc_cnn=False,
                name='TransUNet'):
        assert input_shape[0] % patch_size == 0, "image_size must be a multiple of patch_size"
        x = tf.keras.layers.Input(shape=input_shape)
        
        if hybrid:
            grid_size = grid
            patch_size = input_shape[0] // 16 // grid_size[0]
            if patch_size == 0:
                patch_size = 1

            resnet50v2, features = resnet_embeddings(x, input_shape=input_shape, n_skip=n_skip)
            if freeze_enc_cnn:
                resnet50v2.trainable = False
            y = resnet50v2.get_layer("conv4_block6_preact_relu").output
            x = resnet50v2.input
        else:
            y = x
            features = None
        
        y = tf.keras.layers.Conv2D(
            filters=hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            name="embedding",
            trainable=True
        )(y)
        y = tf.keras.layers.Reshape(
            (y.shape[1] * y.shape[2], hidden_size))(y)
        y = AddPositionEmbs(
            name="Transformer/posembed_input", trainable=True)(y)

        y = tf.keras.layers.Dropout(0.1)(y)

        # Transformer/Encoder
        for n in range(n_layers):
            y, _ = TransformerBlock(
                n_heads=n_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                name=f"Transformer/encoderblock_{n}",
                trainable=True
            )(y)
        y =  tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="Transformer/encoder_norm"
        )(y)
        n_patch_sqrt = int(math.sqrt(y.shape[1]))

        y = tf.keras.layers.Reshape(
            target_shape=[n_patch_sqrt, n_patch_sqrt, hidden_size])(y)

        # Decoder CUP
        if len(decoder_channels):
            y = DecoderCup(decoder_channels=decoder_channels, n_skip=n_skip)(y, features)

        # Segmentation Head
        y = SegmentationHead(num_classes=num_classes, final_act=final_act)(y)

        # Build Model
        self._model =  tf.keras.models.Model(inputs=x, outputs=y, name=name)

    def model(self):
        return self._model



#from resnet_v2 file
def resnet_embeddings(x, input_shape, n_skip=3):
    resnet50v2 = tf.keras.applications.ResNet50V2(weights= None,
                                             include_top=False, 
                                             input_shape=input_shape)
    # resnet50v2.trainable = False
    _ = resnet50v2(x)
    layers = ["conv3_block4_preact_relu",
              "conv2_block3_preact_relu",
              "conv1_conv"]
    features = []
    if n_skip > 0:
        for l in layers:
            features.append(resnet50v2.get_layer(l).output)
    return resnet50v2, features

#from encodef file
class AddPositionEmbs(tf.keras.layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, trainable=True, **kwargs):
        super().__init__(trainable=trainable, **kwargs)
        self.trainable = trainable

    def build(self, input_shape):
        assert (
            len(input_shape) == 3
        ), f"Number of dimensions should be 3, got {len(input_shape)}"
        self.pe = tf.Variable(
            name="pos_embedding",
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=(1, input_shape[1], input_shape[2])
            ),
            dtype="float32",
            trainable=self.trainable,
        )

    def call(self, inputs):
        return inputs + tf.cast(self.pe, dtype=inputs.dtype)
    
class TransformerBlock(tf.keras.layers.Layer):
    """Implements a Transformer block."""

    def __init__(self, *args, n_heads, mlp_dim, dropout, trainable=True, **kwargs):
        super().__init__(*args, trainable=trainable, **kwargs)
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.att = MultiHeadSelfAttention(
            n_heads=self.n_heads,
            name="MultiHeadDotProductAttention_1",
        )
        self.mlpblock = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.mlp_dim,
                    activation="linear",
                    name=f"{self.name}/Dense_0"
                ),
                tf.keras.layers.Lambda(
                    lambda x: tf.keras.activations.gelu(x, approximate=False)
                )
                if hasattr(tf.keras.activations, "gelu")
                else tf.keras.layers.Lambda(
                    lambda x: tfa.activations.gelu(x, approximate=False)
                ),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(
                    input_shape[-1], name=f"{self.name}/Dense_1"),
                tf.keras.layers.Dropout(self.dropout),
            ],
            name="MlpBlock_3",
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_0"
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_2"
        )
        self.dropout = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs, training):
        x = self.layernorm1(inputs)
        x, weights = self.att(x)
        x = self.dropout(x, training=training)
        x = x + inputs
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return x + y, weights

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, *args, trainable=True, n_heads, **kwargs):
        super().__init__(trainable=trainable, *args, **kwargs)
        self.n_heads = n_heads

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        n_heads = self.n_heads
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {n_heads}"
            )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // n_heads
        self.query_dense = tf.keras.layers.Dense(
            hidden_size, name="query")
        self.key_dense = tf.keras.layers.Dense(
            hidden_size, name="key")
        self.value_dense = tf.keras.layers.Dense(
            hidden_size, name="value")
        self.combine_heads = tf.keras.layers.Dense(
            hidden_size, name="out")

    # pylint: disable=no-self-use
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.n_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.hidden_size))
        output = self.combine_heads(concat_attention)
        return output, weights
    
# from decoder file
class Conv2DReLu(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding="same", strides=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
            filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
            padding=self.padding, use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(L2_WEIGHT_DECAY), 
            kernel_initializer="lecun_normal")

        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = tf.nn.relu(x)
        return x

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.conv1 = Conv2DReLu(filters=self.filters, kernel_size=3)
        self.conv2 = Conv2DReLu(filters=self.filters, kernel_size=3)
        self.upsampling = tf.keras.layers.UpSampling2D(
            size=2, interpolation="bilinear")

    def call(self, inputs, skip=None):
        x = self.upsampling(inputs)
        if skip is not None:
            x = tf.concat([x, skip], axis=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DecoderCup(tf.keras.layers.Layer):
    def __init__(self, decoder_channels, n_skip=3, **kwargs):
        super().__init__(**kwargs)
        self.decoder_channels = decoder_channels
        self.n_skip = n_skip

    def build(self, input_shape):
        self.conv_more = Conv2DReLu(filters=512, kernel_size=3)
        self.blocks = [DecoderBlock(filters=out_ch)
                       for out_ch in self.decoder_channels]

    def call(self, hidden_states, features):
        x = self.conv_more(hidden_states)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x

class SegmentationHead(tf.keras.layers.Layer):
    def __init__(self, name="seg_head",
                 num_classes=9,
                 kernel_size=1, 
                 final_act='sigmoid',
                 ** kwargs):
        super(SegmentationHead, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.final_act  = final_act

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
            filters=self.num_classes, kernel_size=self.kernel_size, padding="same",
            kernel_regularizer=tf.keras.regularizers.L2(L2_WEIGHT_DECAY), 
            kernel_initializer=tf.keras.initializers.LecunNormal())
        self.act = tf.keras.layers.Activation(self.final_act)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.act(x)
        return x