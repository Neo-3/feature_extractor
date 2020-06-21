
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np

def extract_image(img):
    model = VGG16(weights='imagenet', include_top=False)
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    
    feat = np.array(model.predict(img_data))
    return feat.flatten()