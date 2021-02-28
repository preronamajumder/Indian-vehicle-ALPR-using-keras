import os
import sys
import cv2
from skimage import measure
from skimage.measure import regionprops
import numpy as np
from keras.models import load_model
import configparser


def get_binary_image(img):
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 29, 2)
    inv_binary = 255 - binary
    return inv_binary

def get_sorted_char_list(cropped_char):
    char_keys = sorted(cropped_char.keys())
    char_list = []
    c = 0
    for x in char_keys:
        char_list.append(cropped_char[x])
        # cv2.imshow("lp", cropped_char[x])
        cv2.imwrite("char"+str(c)+".jpg", cropped_char[x])
        c+=1
        # cv2.waitKey(0)
    return char_list

def character_segmentation(img):
    character_dimensions = (0.3*img.shape[0], 0.9*img.shape[0], 0.02*img.shape[1], 0.5*img.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions

    labelled_img = measure.label(img)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))

    cropped_char = {}
    
    for regions in regionprops(labelled_img):
        y1, x1, y2, x2 = regions.bbox
        region_height = y2 - y1
        region_width = x2 - x1

        if min_height < region_height < max_height and min_width < region_width < max_width \
            and y2 < 0.95*img.shape[0] and y1 > 0.05*img.shape[0] and region_height >= region_width \
            and x2 < 0.98*img.shape[1] and x1 > 0.05*img.shape[1]:

            # print(x1, y1, x2, y2)
            roi = img[max(y1-2, 0):min(y2+2, img.shape[0]), max(x1-2, 0):min(x2+2, img.shape[1])]
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.cvtColor(255-roi, cv2.COLOR_GRAY2RGB)

            xc = x1 + region_width / 2
            # yc = y1 + region_height / 2
            cropped_char[int(xc)] = roi

    #sort characters from left to right
    char_list = get_sorted_char_list(cropped_char)

    return char_list

def get_characters(img):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(e)
    
    img = get_binary_image(img)

    cv2.imshow("lp", img)
    cv2.waitKey(0)

    char_list = character_segmentation(img)
    print(len(char_list))
    return char_list

def load_lp_model(path):
    model = load_model(path)
    model.summary()
    return model

def load_classes(label_path):
    with open(label_path, "r") as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    print(len(classes))
    return classes

def preprocessing(char_list):
    return [char/255 for char in char_list]

def lp_recognition(char_list, model, classes):

    #preprocessing
    char_list = preprocessing(char_list)

    #predict
    y_preds = model.predict(np.array(char_list))
    #print(np.round(y_preds[0], 3))
    y_hat = [np.argmax(y) for y in y_preds]  # list of predictions
    print(np.array(y_hat))

    lp_number = [classes[x] for x in y_hat]
    #TODO: check LP format 
    lp_number = ''.join(lp_number)
    
    return lp_number


def check_config(data):
    if 'model_path' not in data:
        print("Model path not provided")
        sys.exit()
    if 'label_path' not in data:
        print("Label path not provided")
        sys.exit()
    if 'is_folder' not in data:
        print("is_folder not provided")
        sys.exit()
    if data['is_folder'] == 0:
        if 'image_path' not in data:
            print("Image path not provided")
            sys.exit()
    else:
        if 'folder_path' not in data:
            print("Image path not provided")
            sys.exit()
    
def read_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    config.sections()

    data = {}
    for key in config['DEFAULT']:
        if key == "model_file":
            data['model_path'] = config.get('DEFAULT', key)

        if key == "label_file":
            data['label_path'] = config.get('DEFAULT', key)

        if key == "is_folder":
            data['is_folder'] = bool(config.getint('DEFAULT', key))

        if key == "image_path":
            data['image_path'] = config.get('DEFAULT', key)

        if key == "folder_path":
            data['folder_path'] = config.get('DEFAULT', key)
        
        if key == "write_to_csv":
            data['write_to_csv'] = config.get('DEFAULT', key)

    check_config(data)
    return data
    
def main():
    data = read_config("alpr_config.txt")

    #load model
    model = load_lp_model(data['model_path'])
    #load classes
    classes = load_classes(data['label_path'])

    if not data['is_folder']:
        #read license plate image
        img = cv2.imread(data['image_path'])
        #character segmentation
        char_list = get_characters(img)
        #character recognition
        number_plate = lp_recognition(char_list, model, classes)
        print('LP Number: ', number_plate)

    else:
        #read license plate images
        for files in os.listdir(data['folder_path']):
            print(files)
            if files.endswith(".jpg"):
                img = cv2.imread(os.path.join(data['folder_path'],files))
                #character segmentation
                char_list = get_characters(img)
                #character recognition
                number_plate = lp_recognition(char_list, model, classes)

                print('LP Number: ', number_plate)
        

if __name__ == "__main__":
    main()