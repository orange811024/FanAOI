import os, cv2, glob
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.python.keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
import gradio as gr
from keras.models import load_model


images=[]
labels=[]
dict_labels = {"60":0, "100":1}

for folders in glob.glob(os.path.join("D:/Orange/PHOTO/*")):  #修改照片路徑
    print(folders, "圖片讀取中…")
    label = os.path.basename(folders)
    for filename in os.listdir(folders):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            try:
                img_path = os.path.join(folders, filename)
                img = cv2.imread(img_path)
                
                if img is not None:
                    img = cv2.resize(img, dsize=(80, 80))
                    images.append(img)
                    labels.append(dict_labels[label])
                else:
                    print(f"無法讀取圖片: {img_path}")
            except Exception as e:
                print(f"讀取檔案時出錯: {img_path}, 錯誤: {e}")
        else:
            print(f"跳過非圖片檔案: {filename}")

print('圖片數量：{}'.format(len(images)))
print('標籤數量：{}'.format(len(labels)))



train_feature,test_feature,train_label,test_label = \
train_test_split(images,labels,test_size=0.2)
train_feature=np.array(train_feature)
test_feature=np.array(test_feature)
train_label=np.array(train_label)
test_label=np.array(test_label)
train_feature = train_feature/255
test_feature = test_feature/255
train_label = np_utils.to_categorical(train_label)
test_label = np_utils.to_categorical(test_label)

model = Sequential()
model.add(Conv2D(filters=8, kernel_size=(5,5), padding='same',input_shape=(80, 80, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters=16, kernel_size=(5,5),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters=32, kernel_size=(5,5),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=2,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
model.fit(x=train_feature, y=train_label, validation_split=0.2,epochs=20, batch_size=200, verbose=2)
scores = model.evaluate(test_feature, test_label)

print('\n準確率=',scores[1])
model.save('fan_model.h5')



def show_images_labels_predictions(images, labels,predictions,start_id, num=10):
    plt.figure(figsize=(12, 14))
    if num>25: num=25
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[start_id])
        if( len(predictions) > 0 ) :
            title = 'ai = ' + str(predictions[start_id])
            title += (' (o)' if predictions[start_id]== \
            labels[start_id] else ' (x)')
            title += '\nlabel = ' + str(labels[start_id])
        else :
            title = 'label = ' + str(labels[start_id])
        ax.set_title(title,fontsize=12)
        ax.set_xticks([]);ax.set_yticks([])
        start_id+=1
    plt.show()


files = glob.glob("D:/Orange/PHOTO/*.png" )  #修改照片路徑
test_feature=[]
test_label=[]
dict_labels = {"60":0, "100":1}

for file in files:
    img=cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(80,80))
    test_feature.append(img)
    label=file[10:13]
    test_label.append(dict_labels[label])
    test_feature = np.array(test_feature).reshape(len(test_feature),80,80,3).astype('float32')
    test_label = np.array(test_label)

    test_feature_n = test_feature

try:

    model = load_model('/content/fan_model.h5')
    prediction = model.predict(test_feature_n)
    prediction = np.argmax(prediction,axis=1)
    show_images_labels_predictions(test_feature,test_label,prediction,0,len(test_feature))
except:
    print("模型未建立!")



model = load_model("C:/Users/男神/Desktop/Orange/fan_model.h5")  #修改fan_model.h5路徑

def fan60100(image):
    image = np.array(image.resize((80, 80))).astype("float32") / 255.0
    image = image.reshape(1, 80, 80, 3)
    prediction = model.predict(image).tolist()[0]
    class_names = ["60", "100"]
    return {class_names[i]: prediction[i] for i in range(2)}

inp = gr.Image(type="pil")
out = gr.Label(num_top_classes=2, label='預測結果')
grobj = gr.Interface(fn=fan60100, inputs=inp,outputs=out, title="圖片辨識")

grobj.launch(share=True)


# 模組跑完後終端機會提供一組本地應用的地址
# 例:* Running on local URL:  http://127.0.0.1:7860
# 將地址貼至瀏覽器即可打開