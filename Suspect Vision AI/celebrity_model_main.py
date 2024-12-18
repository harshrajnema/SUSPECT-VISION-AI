import tensorflow as tf
# from tensorflow.keras.preprocessing import image <- ERROR [ use 'tf.keras.utils.load_img()' instead of
# 'image.load_img()' ]
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm

# GETTING THE DATA READY (making path file)
import os
actors = os.listdir("dataSet")

filenames2 = []

for actor in actors:
    for file in os.listdir(os.path.join('dataSet',actor)):
        filenames2.append(os.path.join('dataSet',actor,file))

pickle.dump(filenames2,open('filenames2.pkl','wb'))


# TRAINING THE MODEL

filenames = pickle.load(open('filenames2.pkl','rb'))
model =VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

print(model.summary())
# Total params: 23,561,152
# Trainable params: 23,508,032
# Non-trainable params: 53,120

def feature_extractor(img_path,model):
    img =tf.keras.utils.load_img(img_path,target_size=(224,224))
    img_array= tf.keras.utils.load_img.img_to_array(img)
    expanded_img = np.expand_dims(img_array,axis=0)
    preprocessed_img=preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()
    return result

features = []

for file in tqdm(filenames):
    features.append(feature_extractor(file,model))

pickle.dump(features,open('embedding.pkl','wb'))


