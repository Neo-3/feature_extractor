
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

def extract_image(img, model):
    image.LOAD_TRUNCATED_IMAGES = True 
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    
    feat = np.array(model.predict(img_data))
    return feat.flatten()