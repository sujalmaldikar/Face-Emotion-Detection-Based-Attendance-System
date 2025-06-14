from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd 
import numpy as np  


TRAIN = 'images/train'
TEST = 'images/test'

def createdataframe(dir):
    image_path=[]
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir,label)):
            image_path.append(os.path.join(dir,label,imagename))
            labels.append(label)
            print(label,"completed")
    return image_path,labels

train = pd.DataFrame()
train['image'],train['label'] = createdataframe(TRAIN)

print(train)

test = pd.DataFrame()
test['image'],test['label'] = createdataframe(TEST)

print(test['image'])

from tqdm import tqdm
def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image ,color_mode= 'grayscale')
        img = np.array(img)
        features.append(img)

    features = np.array(features)
    features = features.reshape(len(features),48,48,1)
    return features


train_features = extract_features(train['image'])
test_features = extract_features(test['image'])

x_train = train_features/255.0
x_test = test_features/255.0


le = LabelEncoder()
le.fit(train['label'])

y_train = le.transform(train['label'])
y_test = le.transform(test['label'])

y_train = to_categorical(y_train,num_classes=7)
y_test = to_categorical(y_test ,num_classes=7)


model = Sequential()
model.add(Conv2D(128,kernel_size=(3,3),activation='relu', input_shape = (48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7,activation='softmax'))

from keras import metrics

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[metrics.CategoricalAccuracy()])
model.fit(x= x_train,y=y_train, batch_size= 128, epochs=100, validation_data=(x_test,y_test))


model_json = model.to_json()
with open("emotiondetector.json",'w') as json_file:
    json_file.write(model_json)
model.save("emotiondector.h5") 




