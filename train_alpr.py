from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.optimizers import SGD, Adam
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import os
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def load_data(dir):

    image_list = []
    labels = []
    count = 0
    for folder in sorted(os.listdir(dir)):
        print(folder)
        if len(folder) == 1:
            #print('character name: ', folder)
            for image in os.listdir(dir+'/'+folder):
                if image.endswith(".jpg"):
                    #print('character file name: ', image)
                    img = cv2.imread(dir+'/'+folder+'/'+image)/255
                    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
                    image_list.append(img)
                    labels.append(folder)
                    count += 1
                    print(count)

    d = {key: value for value, key in enumerate(set(labels))}
    ls = [d[key] for key in labels]
    ln = np.sort(list(set(labels)))     # all classes
    l = (list(set(ls)))
    class_mapping = {ln[i]: l[i] for i in range(len(ln))}

    print(class_mapping)
    return image_array, labels, class_mapping


def preprocessing(image_array, y):
    import random

    random.seed(0)

    # split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(image_array, y, test_size=0.25, random_state=42, stratify=y)
    train_class = y_train
    test_class = y_test

    # reshaped into original shape (number of images, image height, image width, number of channels)
    X_train = X_train.reshape(len(X_train), X_train[0].shape[0], X_train[0].shape[1], 3)
    X_test = X_test.reshape(len(X_test), X_test[0].shape[0], X_test[0].shape[1], 3)

    # categorical mapping for probabilities
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, X_test, y_train, y_test, train_class, test_class


def cnn(X_train):
    model = Sequential()

    model.add(Conv2D(8, (3, 3), padding='same', input_shape=(X_train[0].shape[0], X_train[0].shape[1], 3),
                     activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))
    # model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                     bias_initializer='glorot_uniform'))
    # model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                     bias_initializer='glorot_uniform'))
    # model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    #model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                     #bias_initializer='glorot_uniform'))
    # model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    #model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                     #bias_initializer='glorot_uniform'))
    # model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    #model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    #model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                     #bias_initializer='glorot_uniform'))
    # model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    #model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                     #bias_initializer='glorot_uniform'))
    # model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    #model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    #model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))
    model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))
    model.add(Dense(34, activation='softmax', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))

    return model


def load_lp_chars(dir):
    image_list = []
    count = 0
    for image in sorted(os.listdir(dir)):
        if image != '.DS_Store':
            if image.endswith('.jpg') or image.endswith('.png'):
                # print('character file name: ', image)
                img = cv2.imread(dir + '/' + image)/255
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
                image_list.append(img)
                count += 1
                print(count)

    image_array = np.array(image_list)

    return image_array

def main():
    dir = './ocr_char'
    image_array, labels, class_mapping = load_data(dir)
    print(len(image_array))
    print(image_array.shape)
    print(len(labels))
    inv_map = {v: k for k, v in class_mapping.items()}
    print(inv_map)

    y = [class_mapping[x] for x in labels]
    print(y)
    X_train, X_test, y_train, y_test, train_class, test_class = preprocessing(image_array, y)
    print(len(class_mapping))

    augment = False

    # model configuration
    model = cnn(X_train)
    model.summary()
    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    #adam = Adam()
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    X_train = X_train.astype('float32')

    if augment:
        datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.07, fill_mode='nearest', rotation_range=30)
        datagen.fit(X_train)
        model.fit_generator(datagen.flow(X_train, y_train, batch_size=20),
                            steps_per_epoch=X_train.shape[0] // 50,
                            epochs=1500,
                            validation_data=(X_test, y_test))
    else:
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=20)

    y_preds = model.predict(X_test)
    np.round(y_preds, 3)

    y_hat = [np.argmax(y) for y in y_preds]  # list of predictions
    np.array(y_hat)

    # evaluation
    acc = accuracy_score(test_class, y_hat)
    print(acc)
    classes = [x for x in class_mapping.keys()]
    print(classification_report(test_class, y_hat, target_names = classes))

    model.save('./Model/lp_recognition_noaugment.h5')
    # model.save_weights('./Model/lp_recognition_weights_noaugment.h5')

if __name__ == '__main__':
    main()