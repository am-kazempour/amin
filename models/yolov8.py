import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from tensorflow.keras.layers import Activation,MaxPooling2D,Conv2D,UpSampling2D,Input,Concatenate,BatchNormalization
from tensorflow.keras.layers import Add,Dense,GlobalAveragePooling2D,Reshape,Lambda,Multiply
import keras_cv
import keras

class yolov8:
    """
    install:
      !pip install --upgrade keras-cv
      !pip install --upgrade keras
    """
    
    def __init__(self,input_shape=(256,256, 3),type="n"):
      self.input_shape = input_shape
      self.input = Input(self.input_shape)
      self.__set_type(type)
      self.load_backbone()
      self.__decoder()
    
    def __set_type(self,type):
      paramet = {
        "n" : [0.33,0.25,2,1]
      }
      self.d,self.w,self.r,self.n = paramet['n']

    def load_backbone(self):
        
        backbone_yolo = keras_cv.models.YOLOV8Backbone.from_preset(
            "yolo_v8_xs_backbone_coco"
        )
        
        intermediate_model = tf.keras.models.clone_model(
          backbone_yolo,
          input_tensors=self.input,
          clone_function=None,
          call_function=None,
          recursive=False,
        )
        
        self.intermediate_model = intermediate_model

    def __decoder(self):
      x = UpSampling2D()(self.intermediate_model.layers[-1].output)
      x = Concatenate()([self.intermediate_model.get_layer("stack3_c2f_output").output,x])
      c2f_2_1 = self.c2f(x,int(540*self.w),self.n)
      x = UpSampling2D()(c2f_2_1)
      x = Concatenate()([self.intermediate_model.get_layer("stack2_c2f_output").output,x])
      c2f_2_2 = self.c2f(x,int(256*self.w),self.n)
      x = self.conv_block(c2f_2_2,filters = int(256*self.w) ,kernel_size = 3,strides=2)
      x = Concatenate()([c2f_2_1,x])
      c2f_3_1 = self.c2f(x,int(512*self.w),3)
      x = self.conv_block(c2f_3_1 , filters = int(512*self.w) ,kernel_size = 3,strides=2)
      x = Concatenate()([self.intermediate_model.layers[-1].output,x])
      c2f_3_2 = self.c2f(x,int(512*self.w),3)
      self.output = self.segmentation_head([c2f_2_2,c2f_3_1,c2f_3_2])
    
    def model(self):
      return keras.Model(inputs = self.input,outputs = [self.output])

    def c2f(self,inputs,filters, n):
      x_1 = self.conv_block(inputs, filters, kernel_size=1 ,  strides=1, padding='same')
      x_2 = self.conv_block(inputs, filters, kernel_size=1 ,  strides=1, padding='same')
      x_b = self.bottle_neck(x_1)
      for i in range(n-1):
        x_b = self.bottle_neck(x_b)
      conc = Concatenate()([x_b, x_2 ])#, axis=3)
      x_out = self.conv_block(conc, filters, kernel_size=1 ,  strides=1, padding='valid')

      return x_out
    
    def conv_block(self,inputs, filters, kernel_size=3, strides=1, padding='same', activation='swish'):
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(inputs)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x

    def SPPF(self,inputs):
      x = self.conv_block(inputs,inputs.shape[-1], kernel_size=1 ,  strides=1, padding='valid')
      pad1 = MaxPooling2D(pool_size=(5, 5),strides=(1, 1), padding="same")(x)
      pad2 = MaxPooling2D(pool_size=(5, 5),strides=(1, 1), padding="same")(pad1)
      pad3 = MaxPooling2D(pool_size=(5, 5),strides=(1, 1), padding="same")(pad2)
      out = Concatenate()([x, pad1, pad2, pad3 ], axis=3)
      out = self.conv_block(out,inputs.shape[-1], kernel_size=1 ,  strides=1, padding='valid')

      return out

    def bottle_neck(self,inputs):
      x = self.conv_block(inputs, inputs.shape[-1], kernel_size=1 , strides=1, padding='valid')
      x = self.conv_block(x, inputs.shape[-1], kernel_size=3 , strides=1, padding='same')
      return x

    def channel_attention(self,x, ratio=8): 
        channel = x.shape[-1] 
        shared_dense_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal') 
        shared_dense_two = Dense(channel, kernel_initializer='he_normal') 
        
        avg_pool = GlobalAveragePooling2D()(x) 
        avg_pool = Reshape((1, 1, channel))(avg_pool) 
        avg_pool = shared_dense_one(avg_pool) 
        avg_pool = shared_dense_two(avg_pool) 
        
        max_pool = Lambda(lambda z: tf.reduce_max(z, axis=[1, 2], keepdims=True))(x) 
        max_pool = shared_dense_one(max_pool) 
        max_pool = shared_dense_two(max_pool) 
        
        cbam_feature = Add()([avg_pool, max_pool]) 
        cbam_feature = Activation('sigmoid')(cbam_feature) 
        
        return Multiply()([x, cbam_feature]) 
 
    def spatial_attention(x): 
        avg_pool = Lambda(lambda z: tf.reduce_mean(z, axis=-1, keepdims=True))(x) 
        max_pool = Lambda(lambda z: tf.reduce_max(z, axis=-1, keepdims=True))(x) 
        concat = Concatenate(axis=-1)([avg_pool, max_pool]) 
        cbam_feature = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid', kernel_initializer='he_normal')(concat) 
        
        return Multiply()([x, cbam_feature]) 
    
    def segmentation_head(self,features): 
        
        channel_att_16x16 = self.channel_attention(features[2]) 
        channel_att_32x32 = self.channel_attention(features[1]) 
        channel_att_64x64 = self.channel_attention(features[0]) 
        upsampled_16x16 = UpSampling2D(size=(2, 2))(channel_att_16x16)   
        combined_32x32 = Concatenate()([channel_att_32x32, upsampled_16x16]) 
        spatial_att_32x32 = self.spatial_attention(combined_32x32) 
        upsampled_32x32 = UpSampling2D(size=(2, 2))(spatial_att_32x32)   
        combined_64x64 = Concatenate()([channel_att_64x64, upsampled_32x32]) 
        spatial_att_64x64 = self.spatial_attention(combined_64x64) 
        upsampled_512x512 = UpSampling2D(size=(8, 8))(spatial_att_64x64)  
        x = Conv2D(256, (3, 3), padding='same', activation='relu')(upsampled_512x512) 
        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x) 
        mask = Conv2D(1, (1, 1), padding='same')(x) 
        mask = Activation('sigmoid')(mask) 
        
        return mask