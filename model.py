import keras
from keras import applications
from keras.models import Model, load_model
from keras.layers import Conv2D, Concatenate, LeakyReLU
from layers import BilinearUpSampling2D
from loss import depth_loss
import os
# suppress verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
'''
TODO:   [ ]Understand Group Convolutions
        [x]Bilinear Upsampling 
        [x]How output sizes are matched in this model
'''
def create_model():



    print('\n\nCreating Model...')

    '''
    Load DenseNet169 with input tensor 640x480x3
    This is the encoder part for our model
    '''
    base_model = applications.DenseNet169(weights = 'imagenet', input_shape = (None, None, 3), include_top = False)
    #base_model.summary()
    '''
    model.layers[-1] returns the last layer of the model
    This is the initial layer for the decoder part
    '''
    base_model_output_shape = base_model.layers[-1].output.shape
    
    # Take the last integer. That's the number of filters
    decode_filters = int(base_model_output_shape[-1])
    for layer in base_model.layers: layer.trainable = True
    
    '''
    BilinearUpSampling2D layers
    TODO: Implement function class in another file.
    '''
    def upsample2d(tensor, filters, name, concat_with):
        upsampled_layer = BilinearUpSampling2D((2, 2), name = name + '_upsampling2d')(tensor)
        # Concatenated skip connection. There are two skip conns: summation and concatenation. You know the difference.
        upsampled_layer = Concatenate(name = name+'_concat')([upsampled_layer, base_model.get_layer(concat_with).output]) 
        upsampled_layer = Conv2D(filters = filters, kernel_size = 3, strides = 1, padding = 'same', name = name+'_conv2A')(upsampled_layer)
        upsampled_layer = LeakyReLU(alpha = 0.2)(upsampled_layer)
        upsampled_layer = Conv2D(filters = filters, kernel_size = 3, strides = 1, padding = 'same', name = name+'_conv2B')(upsampled_layer)
        upsampled_layer = LeakyReLU(alpha = 0.2)(upsampled_layer)
        return upsampled_layer

    decoder = Conv2D(filters = decode_filters, kernel_size = 1, padding = 'SAME',
                    input_shape = base_model_output_shape, name = 'conv2')(base_model.output)
    decoder = upsample2d(decoder, int(decode_filters/2), 'up1', concat_with = 'pool3_pool')
    decoder = upsample2d(decoder, int(decode_filters/4), 'up2', concat_with = 'pool2_pool')
    decoder = upsample2d(decoder, int(decode_filters/8), 'up3', concat_with = 'pool1')
    decoder = upsample2d(decoder, int(decode_filters/16), 'up4', concat_with = 'conv1/relu')
    # Why if_false?
    if False: decoder = upsample2d(decoder, int(decode_filters/32), 'up5', concat_with = 'input_1')

    # Grab depths from these multiple concatenated layers
    conv3 = Conv2D(filters = 1, kernel_size = 3, strides = 1, padding = 'same', name = 'conv3')(decoder)

    # Append inputs and outputs
    model = Model(inputs = base_model.input, outputs = conv3)

    print('\n\nModel Created')
    
    return model