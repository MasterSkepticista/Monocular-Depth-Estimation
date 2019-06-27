import os
from model import create_model
import numpy as np
import keras
from data import get_nyu_train_test_data, load_images, predict, show_images, to_multichannel
from loss import depth_loss
# suppress verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'


#------------
# Constants/HP
#------------
BATCH_SIZE = 6
lr = 0.0001
EPOCHS = 1
#------------Test images snippet
import glob
image_list = glob.glob('*.jpg')
test_images = load_images(image_list)
print(test_images.shape)
show_images(test_images)
#------------
'''
Create Model with Decoder
'''
model = create_model()

'''
Create Train and Test Generators
Returns Data_generator objects for Keras
'''
train_generator, test_generator = get_nyu_train_test_data(BATCH_SIZE)

print('\n\nGenerators Ready:', train_generator, test_generator)

optimizer = keras.optimizers.Adam(lr = lr)

model.compile(loss = depth_loss, optimizer = optimizer)
print('\n\nModel Compiled. Ready to train.')

model.fit_generator(train_generator, validation_data = test_generator, epochs = EPOCHS, shuffle = True)

print('Finished Training. Running Inference')


predictions = predict(model, test_images)

print(predictions.shape)
outputs = to_multichannel(predictions)
show_images(outputs, save = True)
