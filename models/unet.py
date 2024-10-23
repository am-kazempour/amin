import tensorflow as tf
from tensorflow.keras import layers, Model


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
        self.__architecture()

    def model(self):
        return Model(inputs = self.input,outputs = [self.output])

    def __architecture(self):
        if self.encoder_num == 1:
            output, c4, c3, c2, c1= self.__encoder(self.input)
        else:
            output1, c41, c31, c21, c11= self.__encoder(self.input[:,:,:,:2])
            output2, c42, c32, c22, c12= self.__encoder(self.input[:,:,:,2:])
            output = layers.concatenate([output1,output2])
            c4 = layers.concatenate([c41,c42])
            c3 = layers.concatenate([c31,c32])
            c2 = layers.concatenate([c21,c22])
            c1 = layers.concatenate([c11,c12])
        
        output = self.__bottleneck(output)
        x = self.__decoder(output, c4, c3, c2, c1)

        x = layers.Conv2D(self.class_num, (1, 1))(x)
        if self.class_num == 1:
            self.output = layers.Activation('sigmoid')(x)
        else:
            self.output = layers.Activation('softmax')(x)

    def __encoder(self,input):

        c1 = self.conv_block(input,self.num_filters)
        x = layers.MaxPooling2D((2, 2))(c1)

        c2 = self.conv_block(x, self.num_filters*2)
        x = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = self.conv_block(x, self.num_filters*4)
        x = layers.MaxPooling2D((2, 2))(c3)
        
        c4 = self.conv_block(x, self.num_filters*8)
        x = layers.MaxPooling2D((2, 2))(c4)
        
        return x, c4, c3, c2, c1

    def __bottleneck(self,input,activation='relu'):
        x = layers.Conv2D(self.num_filters*16, (3, 3), padding='same')(input)
        if self.batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Conv2D(self.num_filters*16, (3, 3),  padding='same')(x)
        if self.batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        return x

    def __decoder(self, input, c4, c3, c2, c1):
        
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

