from PIL import Image
import os, glob, sys, numpy as np
from sklearn.model_selection import train_test_split

img_dir = './image'
categories = ['cat', 'dog']
np_classes = len(categories)

image_w = 64
image_h = 64

pixel = image_h * image_w

x = []
y = []

for idx, category in enumerate(categories):
    img_dir_detail = img_dir + '/' + category
    files = glob.glob(img_dir_detail+'/*.jpg')

    for i, f in enumerate(files):
        try:
            img = Image.open(f)
            img = img.convert('RGB')
            img = img.resize((image_w, image_h))
            data = np.asarray(img)
            # y는 0 아니면 1이니까 idx값으로 넣는다.
            x.append(data)
            y.append(idx)
            if i % 300 == 0:
                print(category, ' : ', f)
        except:
            print(category, str(i) + " 번째에서 에러")

X = np.array(x)
Y = np.array(y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

xy = (X_train, X_test, Y_train, Y_test)
np.save('./numpy_data/binary_image_data.npy', xy)