#นำเข้าไลบรารี่เตรียมตัวสำหรับการใช้งานนะครับน้องๆ
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras import callbacks as cb
import os, json, math, librosa
import IPython.display as ipd
import librosa.display as dis
import librosa
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, MaxPooling2D, Input, Activation

import sklearn.model_selection as sk
from sklearn.model_selection import train_test_split

import librosa
import numpy as np
import matplotlib.pyplot as plt

file = "C:\\Users\\mawza\\Desktop\\WorkHere\\warat\\MainOurAi2\\genresong\\Country\\country001.wav"
print(file)

# waveform
signal, sr = librosa.load(file, sr=22050)
plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(signal)) / sr, signal)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform")
plt.show()

# fft -> spectrum
fft = np.fft.fft(signal)

magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))
left_freq = frequency[: int(len(frequency) / 2)]
left_mag = magnitude[: int(len(magnitude) / 2)]

plt.plot(left_freq, left_mag, color="b")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT Spectrum")
plt.show()

# fft
n_fft = 2048  # window when considering performing a single fft
hop_length = 512  # amount shifting after each transform

stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)

librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency (Hz)")
plt.title("Spectrogram")
plt.colorbar(format="%+2.0f dB")
plt.show()

# mfcc
MFCCs = librosa.feature.mfcc(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")  
plt.ylabel("MFCC Coefficients")
plt.title("MFCCs")
plt.colorbar()
plt.show()

import os
MUSIC = r'C:\\Users\\mawza\\Desktop\\WorkHere\\warat\\MainOurAi2\\genresong' 
music_dataset = []  # สร้าง music_dataset เพื่อเก็บเส้นทางของไฟล์เสียงดนตรีแต่ละไฟล์
genre_target = []  # สร้าง genre_target เพื่อเก็บแนวเพลงของแต่ละไฟล์

for root, dirs, files in os.walk(MUSIC):
    for name in files:
        filename = os.path.join(root, name)
        
        # ตรวจสอบว่าแนวเพลงไม่ใช่ 'genresong' ก่อนที่จะเพิ่มลงในรายการ genre_target
        if os.path.basename(os.path.dirname(filename)) != 'genresong':
            music_dataset.append(filename)

            # แยกชื่อแนวเพลงจากเส้นทางไฟล์
            genre = os.path.basename(os.path.dirname(filename))
            genre_target.append(genre)

print("genres in dataset: ", set(genre_target))

audio_path = music_dataset[60]#เลือกลำดับเพลงในนี้
x , sr = librosa.load(audio_path)
librosa.load(audio_path, sr=None)

ipd.Audio(audio_path)

# Visualizing Audio File as a waveform
plt.figure(figsize=(16, 5))
# librosa.display.waveplot(x, sr=sr)
# Visualizing audio file as a spectogram
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.title('Spectogram')
plt.colorbar()
file_location = audio_path
y, sr = librosa.load(file_location)
melSpec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
melSpec_dB = librosa.power_to_db(melSpec, ref=np.max)
plt.figure(figsize=(10, 5))
librosa.display.specshow(melSpec_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
plt.colorbar(format='%+1.0f dB')
plt.title("MelSpectrogram")
plt.tight_layout()
plt.show()


DATASET_PATH = r'C:\\Users\\mawza\\Desktop\\WorkHere\\warat\\MainOurAi2\\genresong'
JSON_PATH = "features_data.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }
#กำหนดพารามิเตอร์ต่าง ๆ ที่เกี่ยวข้องกับการสร้าง MFCC เช่น SAMPLE_RATE, TRACK_DURATION, และ SAMPLES_PER_TRACK
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            # ดึงชื่อโฟลเดอร์สุดท้ายจากเส้นทางไฟล์
            semantic_label = os.path.basename(dirpath)
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                for d in range(num_segments):
                    start = samples_per_segment * d
                    finish = start + samples_per_segment
                    mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
save_mfcc(DATASET_PATH, JSON_PATH, num_segments=15)#ใช้ฟังก์ชัน save_mfcc เพื่อดึงคุณลักษณะ MFCC จากไฟล์เสียงดนตรีและบันทึกข้อมูลลงใน JSON ไฟล์ เพื่อนำมาใช้ในการเทรนโมเดล
print("process finished")

DATA_PATH = "./features_data.json"

def load_data(data_path): #ฟังก์ชัน load_data เพื่อโหลดข้อมูล MFCC จาก JSON ไฟล์ที่สร้างไว้และคืนค่าข้อมูลที่โหลดในรูปแบบของ Numpy arrays (X, y) และรายชื่อแนวเพลง (z)
 
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    z = np.array(data['mapping'])
    return X, y, z


def plot_history(history): #แสดงค่าเป็นกราฟในแต่ละครั้งที่เทรนเพื่อหาประมวลความถูกต้อง
    fig, axs = plt.subplots(2)
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")
    
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size): #ใช้ prepare_datasets เพื่อแบ่งข้อมูลเป็นชุดการฝึก (training), ชุดการตรวจสอบ (validation), และชุดทดสอบ (test)
    X, y, z = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle = True,random_state =42)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=1, shuffle = True, random_state = 42)

    #ข้อมูลถูกแปลงให้มีขนาดเหมาะสมสำหรับการนำเข้าโมเดล CNN โดยเพิ่มมิติในชุดข้อมูลด้วย np.newaxis
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test, z

#ใช้ build_model เพื่อสร้างโมเดล CNN ที่ประกอบด้วยชั้น Convolutional, BatchNormalization, MaxPooling, Flatten, และ Dense
def build_model(input_shape):
    model = Sequential()
#     1st conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape, kernel_initializer = 'he_normal'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', kernel_initializer = 'he_normal'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', kernel_initializer = 'he_normal'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer = 'he_normal'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu', kernel_initializer = 'he_normal'))
    model.add(keras.layers.Dropout(0.5))
    # output layer
    model.add(keras.layers.Dense(4, activation='softmax')) #ชั้นสุดท้ายมี 4 หน่วยตามชนิดของเพลงและฟังก์ชัน activation เป็น softmax เพื่อใช้ในการจำแนกแนวเพลง
    return model

    #ใช้ predict เพื่อทำนายแนวเพลงของตัวอย่าง
def predict(model, X, y,z):
    
    X = X[np.newaxis, ...]
    prediction = model.predict(X)
    predicted_index = np.argmax(prediction, axis=1)
    target = z[y]
    predicted = z[predicted_index]

    print("Target: {}, Predicted label: {}".format(target, predicted))

DATA_PATH = "./features_data.json"

def load_data(data_path): #ฟังก์ชัน load_data เพื่อโหลดข้อมูล MFCC จาก JSON ไฟล์ที่สร้างไว้และคืนค่าข้อมูลที่โหลดในรูปแบบของ Numpy arrays (X, y) และรายชื่อแนวเพลง (z)
 
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    z = np.array(data['mapping'])
    return X, y, z


def plot_history(history): #แสดงค่าเป็นกราฟในแต่ละครั้งที่เทรนเพื่อหาประมวลความถูกต้อง
    fig, axs = plt.subplots(2)
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")
    
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size): #ใช้ prepare_datasets เพื่อแบ่งข้อมูลเป็นชุดการฝึก (training), ชุดการตรวจสอบ (validation), และชุดทดสอบ (test)
    X, y, z = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle = True,random_state =42)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=1, shuffle = True, random_state = 42)

    #ข้อมูลถูกแปลงให้มีขนาดเหมาะสมสำหรับการนำเข้าโมเดล CNN โดยเพิ่มมิติในชุดข้อมูลด้วย np.newaxis
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test, z

#ใช้ build_model เพื่อสร้างโมเดล CNN ที่ประกอบด้วยชั้น Convolutional, BatchNormalization, MaxPooling, Flatten, และ Dense
def build_model(input_shape):
    model = Sequential()
#     1st conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape, kernel_initializer = 'he_normal'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', kernel_initializer = 'he_normal'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', kernel_initializer = 'he_normal'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu', kernel_initializer = 'he_normal'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu', kernel_initializer = 'he_normal'))
    model.add(keras.layers.Dropout(0.5))
    # output layer
    model.add(keras.layers.Dense(4, activation='softmax')) #ชั้นสุดท้ายมี 4 หน่วยตามชนิดของเพลงและฟังก์ชัน activation เป็น softmax เพื่อใช้ในการจำแนกแนวเพลง
    return model

    #ใช้ predict เพื่อทำนายแนวเพลงของตัวอย่าง
def predict(model, X, y,z):
    
    X = X[np.newaxis, ...]
    prediction = model.predict(X)
    predicted_index = np.argmax(prediction, axis=1)
    target = z[y]
    predicted = z[predicted_index]

    print("Target: {}, Predicted label: {}".format(target, predicted))

# บันทึกโมเดลหลังจากเทรนเสร็จสิ้น
model.save('my_model.keras')

# pick a sample to predict from the test set
X_to_predict = X_test[165]
y_to_predict = y_test[165]
# predict sample
print(predict(model, X_to_predict, y_to_predict,z))

model.evaluate(X_test, y_test)
#model.evaluate เพื่อประเมินประสิทธิภาพของโมเดลบนชุดทดสอบ
