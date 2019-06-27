from zipfile import ZipFile
from augment import BasicPolicy
from keras.utils import Sequence
from PIL import Image
import numpy as np
from io import BytesIO

def DepthNorm(depth_image, MaxDepth = 1000.0):
    return MaxDepth/depth_image

def nyu_resize(img, resolution = 480, padding = 6):
    from skimage.transform import resize
    return resize(img, (resolution, int(resolution *4/3)), preserve_range=True, anti_aliasing=True, mode='reflect')

def extract_zip(input_zip):
    input_zip = ZipFile(input_zip)

    return {name: input_zip.read(name) for name in input_zip.namelist() }

def get_nyu_data(batch_size, nyu_data_zip = 'nyu_data.zip'):
    # Extract zip file
    # Returns directory and file dictionary
    dictionary = extract_zip(nyu_data_zip)

    nyu2_train = list((row.split(',') for row in (dictionary['data/nyu2_train.csv']).decode('utf-8').split('\n') if len(row)>0))
    nyu2_test = list((row.split(',') for row in (dictionary['data/nyu2_test.csv']).decode('utf-8').split('\n') if len(row) > 0))
    print('Got the file list')

    shape_rgb = (batch_size, 480, 640, 3)
    shape_depth = (batch_size, 240, 320, 1)

    '''
    To debug, set this flag True to only pass 10 images list instead of all
    Return dictionary, RGB and depth shapes, and the list of files pointing to gt
    '''
    
    if False:
        nyu2_train = nyu2_train[:2000]
        nyu2_test = nyu2_test[:2000]
        
    return dictionary, nyu2_train, nyu2_test, shape_rgb, shape_depth


def get_nyu_train_test_data(batch_size):
    data, nyu2_train, nyu2_test, shape_rgb, shape_depth = get_nyu_data(batch_size)

    '''
    Call a sequence class from Keras generator which will take this list
    of Images (RGB + Depth) and returns a keras data generator object.
    It should Augment images as well
    '''
    train_generator = NYU_BasicAugmentRGBSequence(data, nyu2_train, batch_size = batch_size, shape_rgb = shape_rgb, shape_depth = shape_depth)
    test_generator = NYU_BasicRGBSequence(data, nyu2_test, batch_size = batch_size, shape_rgb = shape_rgb, shape_depth = shape_depth)

    return train_generator, test_generator

class NYU_BasicAugmentRGBSequence(Sequence):
    def __init__(self, data, dataset_list, batch_size, shape_rgb, shape_depth, is_flip = False, is_addnoise = False, is_erase = False):
        self.data = data
        self.dataset_list = dataset_list
        self.batch_size = batch_size
        self.shape_depth = shape_depth
        self.shape_rgb = shape_rgb
        self.policy = BasicPolicy(color_change_ratio=0.5, mirror_ratio=0.5, flip_ratio=0.0 if not is_flip else 0.2,
                            add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.MaxDepth = 1000.0

        from sklearn.utils import shuffle
        self.dataset_list = shuffle(self.dataset_list, random_state = 0)

        self.N = len(self.dataset_list)

    '''
    This function defines the number of mini-batches.
    Or batches per epoch.
    floor(total_samples/batch_size)
    '''
    def __len__(self):  return int(np.floor(self.N / self.batch_size))
    '''
    Consider apply policy flag toggle if something goes wrong (which shouldn't actually)
    '''
    def __getitem__(self, idx, is_apply_policy = True):
        '''
        Create empty numpy arrays of required shapes
        '''
        batch_x, batch_y = np.zeros(self.shape_rgb), np.zeros(self.shape_depth)

        '''
        Read this as:
        for i in every image of batch, augment i
        idx refers to minibatch index, of all batches
        '''
        for i in range(batch_x.shape[0]):
            index = min((self.batch_size * idx) + i, self.N-1)
            # Get the tuple containing path of RGB and Depth
            sample = self.dataset_list[index]

            # Read Images from list->bytes
            rgb_image = Image.open(BytesIO(self.data[sample[0]]))
            depth_image = Image.open(BytesIO(self.data[sample[1]]))

            # Reshape in dims
            rgb_image = np.asarray(rgb_image).reshape(480, 640, 3)
            depth_image = np.asarray(depth_image).reshape(480, 640, 1)

            # clip and norm
            rgb_image = np.clip(rgb_image/255, 0, 1)
            depth_image = np.clip(depth_image/255 * self.MaxDepth, 0, self.MaxDepth)
            depth_image_norm = DepthNorm(depth_image, self.MaxDepth)

            # Place into empty arrays after resize. What?
            # Maybe to increase code flexibility.
            batch_x[i] = nyu_resize(rgb_image, 480)
            batch_y[i] = nyu_resize(depth_image_norm, 240)

            # Doesn't really matter. For simplicity sake I'm not gonna use this
            if is_apply_policy: batch_x[i], batch_y[i] = self.policy(batch_x[i], batch_y[i])

        return batch_x, batch_y

class NYU_BasicRGBSequence(Sequence):
    '''
    Keep in mind, no Augment here
    '''
    def __init__(self, data, dataset_list, batch_size, shape_rgb, shape_depth):
        self.data = data
        self.dataset_list = dataset_list
        self.batch_size = batch_size
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.N = len(self.dataset_list)
        self.MaxDepth = 1000.0

    def __len__(self):
        return int(np.floor(self.N/self.batch_size))


    def __getitem__(self, idx):

        # Create empty holders
        batch_x, batch_y = np.zeros(self.shape_rgb), np.zeros(self.shape_depth)

        for i in range(self.batch_size):

            # index is much like a global ptr
            index = min(idx * self.batch_size + i, self.N - 1)
            sample = self.dataset_list[index]

            rgb_image = Image.open(BytesIO(self.data[sample[0]]))
            depth_image = Image.open(BytesIO(self.data[sample[1]]))

            rgb_image = np.asarray(rgb_image).reshape(480, 640, 3)
            depth_image = np.asarray(depth_image, dtype = np.float32).reshape(480, 640, 1)

            rgb_image = np.clip(rgb_image/255, 0, 1)
            # Why?
            depth_image = depth_image.copy().astype(float) / 10.0

            depth_norm = DepthNorm(depth_image, self.MaxDepth)

            batch_x[i] = nyu_resize(rgb_image, 480)
            batch_y[i] = nyu_resize(depth_norm, 240)

        return batch_x, batch_y
        

def load_images(image_list):
    loaded_images = []
    for file in image_list:
        x = np.clip(np.array(Image.open(file), dtype = float) / 255, 0, 1)
        loaded_images.append(x)
    
    return np.stack(loaded_images, axis = 0)

def predict(model, images, minDepth = 10.0, maxDepth = 1000.0, batch_size = 2):
    outputs = []
    outputs = model.predict(images, batch_size=batch_size)

    return np.clip(DepthNorm(outputs, MaxDepth = 1000), minDepth, maxDepth) / maxDepth

def show_images(images, save = False):
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(2, 3)
    axis[0, 0].imshow(images[0][:,:,0], cmap = 'plasma')
    axis[0, 1].imshow(images[1][:,:,0], cmap = 'plasma')
    axis[0, 2].imshow(images[2][:,:,0], cmap = 'plasma')
    axis[1, 0].imshow(images[3][:,:,0], cmap = 'plasma')
    axis[1, 1].imshow(images[4][:,:,0], cmap = 'plasma')
    axis[1, 2].imshow(images[5][:,:,0], cmap = 'plasma') 

    plt.show()
    if save:
        plt.savefig('result.png')

def to_multichannel(inputs):
    if inputs.shape[3] > 1:
        print('No expansion needed')
        return inputs
    elif inputs.shape[3] == 1:
        outputs = []
        for i in range(inputs.shape[0]):
            channel = inputs[i][:,:,0]
            outputs.append(np.stack((channel, channel, channel), axis = 2))
        return np.asarray(outputs)
