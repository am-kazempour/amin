import tensorflow as tf
from tensorflow.keras import layers, Model

from .unet import Unet


class bos_seg_Unet(Unet):

    def __init__(self, input_shape=..., num_filters=64, class_num=1, batch_norm=True, encoder_num=1,decoder_num=3):
        self.decoder_num = decoder_num
        super().__init__(input_shape, num_filters, class_num, batch_norm, encoder_num)

    def _architecture(self):
        output, c4, c3, c2, c1= self._encoder(self.input)
        output = self._bottleneck(output)
        x = []
        for _ in range(self.decoder_num):
            x.append(self._head(self._decoder(output, c4, c3, c2, c1)))
        self.output = layers.concatenate(x)
