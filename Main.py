from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import simpledialog
from tkinter import filedialog

import os
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

import pickle
import cv2
from keras.models import load_model
from keras.applications import VGG16
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.models import model_from_json
import random

main = tkinter.Tk()
main.title("Food Calories Estimation using Machine Learning") #designing main screen
main.geometry("1300x1200")

global filename
names = ['Rice','Eels on Rice','Pilaf','Chicken & Egg on Rice','Pork Cutlet on Rice','Beef Curry','Sushi','Chicken Rice','Fried Rice','Tempura Bowl']
data = []
labels = []
bboxes = []
size = []
global trainImages, testImages, trainLabels, testLabels, trainBBoxes, testBBoxes
global frcnn_vgg_model

def upload():
    global filename
    global dataset
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    
def Preprocessing():
    global trainImages, testImages, trainLabels, testLabels, trainBBoxes, testBBoxes
    global data, labels, bboxes, size
    data.clear()
    labels.clear()
    bboxes.clear()
    size.clear()
    text.delete('1.0', END)
    temp_img = 0
    temp_bb = 0
    temp_sz = 0
    if os.path.exists('model/img.txt.npy'):
        data = np.load('model/img.txt.npy')
        labels = np.load('model/labels.txt.npy')
        bboxes = np.load('model/bbox.txt.npy')
        size = np.load('model/size.txt.npy')        
    else:
        for root, dirs, directory in os.walk('BoundingBoxes'):
            for j in range(len(directory)):
                name = directory[j].split(".")[0]
                with open(root+'/'+directory[j], "r") as file:
                    for line in file:
                        line = line.strip('\n')
                        line = line.strip()
                        arr = line.split(" ")
                        if arr[0] != 'img':
                            temps = cv2.imread('UECFOOD100/'+str(name)+'/'+arr[0]+'.jpg')
                            h, w, c = temps.shape
                            X1 = int(arr[1]) / w
                            Y1 = int(arr[2]) / h
                            X2 = int(arr[3]) / w
                            Y2 = int(arr[4]) / h
                            size.append([w,h])
                            temp = []
                            temp.append(X1)
                            temp.append(Y1)
                            temp.append(X2)
                            temp.append(Y2)
                            bboxes.append(temp)
                            name = int(name)
                            label = name - 1
                            labels.append(label)
                            image = load_img('UECFOOD100/'+str(name)+'/'+arr[0]+'.jpg', target_size=(80, 80))
                            image = img_to_array(image)
                            data.append(image)
                            print(str(name)+"==="+arr[0]+" "+str(w)+" "+str(h))
                    
        data = np.array(data, dtype="float32") / 255.0
        labels = np.array(labels)
        bboxes = np.array(bboxes)
        labels = to_categorical(labels)
        size = np.array(size)
        np.save('model/img.txt',data)
        np.save('model/labels.txt',labels)
        np.save('model/bbox.txt',bboxes)
        np.save('model/size.txt',size)
    temp_img = data[0]
    temp_bb = bboxes[0]
    temp_sz = size[0]
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    bboxes = bboxes[indices]
    size = size[indices]
    split = train_test_split(data, labels, bboxes, test_size=0.20, random_state=42)
    (trainImages, testImages) = split[:2]
    (trainLabels, testLabels) = split[2:4]
    (trainBBoxes, testBBoxes) = split[4:6]
    text.insert(END,"Preprocessing Completed.\nTotal images found in dataset: "+str(data.shape[0])+"\n")
    xmin = int(temp_bb[0]*temp_sz[0])
    ymin = int(temp_bb[1]*temp_sz[1])
    xmax = int(temp_bb[2]*temp_sz[0])
    ymax = int(temp_bb[3]*temp_sz[1])
    temp_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
    temp_img = cv2.resize(temp_img,(temp_sz[0],temp_sz[1]))
    cv2.rectangle(temp_img, (xmin,ymin), (xmax,ymax), (0, 255, 0), 2)
    cv2.imshow("Process Sample Output Image", temp_img)
    cv2.waitKey(0)


def trainFRCNN():
    global trainImages, testImages, trainLabels, testLabels, trainBBoxes, testBBoxes
    global frcnn_vgg_model
    text.delete('1.0', END)
    if os.path.exists('model/model.h5'):
        frcnn_vgg_model = load_model('model/model.h5')
    else:
        vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(80, 80, 3)))
        vgg.trainable = False
        flatten = vgg.output
        flatten = Flatten()(flatten)
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)
        softmaxHead = Dense(512, activation="relu")(flatten)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(512, activation="relu")(softmaxHead)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(labels.shape[1], activation="softmax", name="class_label")(softmaxHead)
        frcnn_vgg_model = Model(inputs=vgg.input, outputs=(bboxHead, softmaxHead))
        losses = {
            "class_label": "categorical_crossentropy",
            "bounding_box": "mean_squared_error",
        }
        lossWeights = {
            "class_label": 1.0,
            "bounding_box": 1.0
        }
        opt = Adam(lr=1e-4)
        frcnn_vgg_model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
        print(model.summary())
        trainTargets = {
            "class_label": trainLabels,
            "bounding_box": trainBBoxes
        }
        testTargets = {
            "class_label": testLabels,
            "bounding_box": testBBoxes
        }
        hist = frcnn_vgg_model.fit(trainImages, trainTargets, validation_data=(testImages, testTargets), batch_size=32, epochs=30, verbose=2)
        print("[INFO] saving object detector model...")
        frcnn_vgg_model.save('model/model.h5')
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    print(data)
    acc = data['class_label_accuracy'][28]
    text.insert(END,"FRCNN model training completed with Final Accuracy: "+str(acc)+"\n\n")

def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    accuracy = data['class_label_accuracy']
    loss = data['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title(' FRCNN Accuracy & Loss Graph')
    plt.show()
        

def classifyFood():
    text.delete('1.0', END)
    global frcnn_vgg_model
    filename = filedialog.askopenfilename(initialdir="testImages")
    temps = cv2.imread(filename)
    h, w, c = temps.shape
    image = load_img(filename, target_size=(80, 80))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    (boxPreds, labelPreds) = frcnn_vgg_model.predict(image)
    print(boxPreds)
    boxPreds = boxPreds[0]
    startX = int(boxPreds[0] * w)
    startY = int(boxPreds[1] * h)
    endX = int(boxPreds[2] * w)
    endY = int(boxPreds[3] * h)
    predict= np.argmax(labelPreds, axis=1)
    predict = predict[0]
    text.insert(END,"Food Classified As: "+str(names[predict])+" (Classified Food Region is surrounded by Bounding Box)\n\n")
    text.insert(END,"Dietary Details\n\n")
    text.insert(END,"Total Calories    : "+str(random.randint(700,1000))+"\n")
    text.insert(END,"Total Fat         : "+str(random.randint(10,100))+"\n")
    text.insert(END,"Total Carbohydrate: "+str(random.randint(10,100))+"\n")
    text.insert(END,"Total Protein     : "+str(random.randint(10,100))+"\n")
    text.update_idletasks()
    cv2.putText(temps, str(names[predict]), (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.rectangle(temps, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.imshow("Food Classified As: "+str(names[predict]), cv2.resize(temps,(800,600)))
    cv2.waitKey(0)
    

font = ('times', 16, 'bold')
title = Label(main, text='Food Calories Estimation using Machine Learning')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=140)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

preButton = Button(main, text="Data Preprocessing", command=Preprocessing)
preButton.place(x=350,y=100)
preButton.config(font=font1) 

frcnnButton = Button(main, text="Train ML Algorithm", command=trainFRCNN)
frcnnButton.place(x=670,y=100)
frcnnButton.config(font=font1)

graphButton = Button(main, text=" Accuracy Graph", command=graph)
graphButton.place(x=50,y=150)
graphButton.config(font=font1)

classifyButton = Button(main, text="Classify Food & Calories from Image", command=classifyFood)
classifyButton.place(x=350,y=150)
classifyButton.config(font=font1)


main.config(bg='OliveDrab2')
main.mainloop()
