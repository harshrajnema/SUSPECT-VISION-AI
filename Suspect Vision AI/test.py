# Extract its features
# find the cosine distance of current image  with all the 8655 features
# recommend that image

import pickle
import cv2
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
from PIL import Image

feature_list= np.array(pickle.load(open('embedding.pkl','rb')))
filenames =pickle.load(open('filenames2.pkl','rb'))

model =VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

detector =MTCNN()


# load image ---> face detection

sample_img =cv2.imread('sample/anuskha_dup.jpg')
result=detector.detect_faces(sample_img)

x,y,width,height=result[0] ['box']

face = sample_img[y:y+height,x:x+width]
# cv2.imshow('output',face)
# cv2.waitKey(0)



# Extract its features------------>

image=Image.fromarray(face)
image=image.resize((224,224))

face_array =np.asarray(image)
face_array =face_array.astype('float32')

expanded_img= np.expand_dims(face_array,axis=0)
preprocessed_img= preprocess_input(expanded_img)
result=model.predict(preprocessed_img).flatten()
# print(result)
# print(result.shape)


# find the cosine distance of current image  with all the 8655 features
# similarity.append(cosine_similarity(result.reshape(1, -1), feature_list[0].reshape(1, -1))[0][0])
similarity =[]
for i in range(len(feature_list)):
    similarity.append(cosine_similarity(result.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

# Next is to Add the Index to our dataSet--------------->

# print(len(similarity)) #Output ~8548
# print(list(enumerate(similarity)))


# Here we basically Added the Index to our dataSet, ie... All the Items [Images] are represented as a List inside
# tuple { List[  (Tuple1),(Tuple2),(Tuple3)----(Tuple8548)  ] which shows the similarity ratio with respect to our
# or on the basis sample image.
#
# for example----> [(0, 0.24629338), (1, 0.26207998), (2, 0.25299078), (3, 0.2566548), (4, 0.34692633),
# (5, 0.2606871), (6, 0.30901486), (7, 0.29185385), (8, 0.25456893), (9, 0.3448801), (10, 0.28034654), (11,
# 0.21142837), (12, 0.37018353), (13, 0.27707186), (14, 0.32444447), (15, 0.37984475), (16, 0.35782757), (17,
# 0.3169791), (18, 0.3643373), (19, 0.21973065), (20, 0.23384741), (21, 0.27034163), (22, 0.2964331), (23,
# 0.27749312), (24, 0.31157863), (25, 0.27568263), (26, 0.2838496), (27, 0.25919712), (28, 0.23285016), (29,
# 0.21124211), (30, 0.26180682), (31, 0.3080216), (32, 0.37446553), (33, 0.26485124)
# -------------------------------------, (8546, 0.1935102), (8547, 0.24548814)]

# Next Step is to sort the dataSet in descending Order or reverse order--------------->

# print(sorted(list(enumerate(similarity)),reverse=True, key=lambda  x:x[1]))

# [(6150, 0.6651572), (6149, 0.6231524), (6134, 0.58989525), (6188, 0.5860761), (6191, 0.5851657), (6174, 0.5753732),
# (6190, 0.57495457), (6143, 0.57493323), (6187, 0.57238024), (6198, 0.57140124), (6137, 0.56257945), (6158,
# 0.5546802), (6180, 0.55408955), (6169, 0.5497228), (6144, 0.5476291), (6141, 0.54432), (6155, 0.535238), (6182,
# 0.53484726), (5817, 0.53474915), (5824, 0.5323274), (6163, 0.5313468), (5852, 0.5305004), (6147, 0.5299679), (6132,
# 0.5294571), (6172, 0.52748966), (6175, 0.52554405), (6181, 0.52518237), (6161, 0.5239935), (5842, 0.5239625),
# (6159, 0.52114004), (6168, 0.5195234), (6171, 0.5170433), (6165, 0.51624453), (7288, 0.5116451), (6148,
# 0.51117086), (5838, 0.51088846), (5897, 0.51081675), (5815, 0.51075244), (6176, 0.51073873), (7279, 0.5097378),
# (6145, 0.5093932), (5826, 0.5084725), (6173, 0.506444), (559, 0.5060237), (6192, 0.5035784), (5820, 0.5032397),
# (622, 0.50276035), (6139, 0.50067854), (5848, 0.500442), (6157, 0.49967125), (6131, 0.49956295), (6177,
# 0.49777406), (5865, 0.49675196), (6162, 0.49426943), (6140, 0.49299425), (5833, 0.4922277), (5818, 0.4898559),
# (5853, 0.48956868), (6154, 0.4885317), (7275, 0.48552638), (6146, 0.48464152), --------------------------->,
# (155, 0.051606417), (156, 0.040701065)]

# Next Step To display that image whose Similarity ratio is Maximum as compare to all the Image Present in the
# DataSet--------------->

index_position=sorted(list(enumerate(similarity)),reverse=True, key=lambda  x:x[1]) [0][0]
temp_img=cv2.imread(filenames[index_position])
cv2.imshow('output',temp_img)
cv2.waitKey(0)


# RECOMMEND  THAT IMAGE---------------->



