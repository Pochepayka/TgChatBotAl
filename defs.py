import cv2
import os
import tensorflow as tf
from collections import Counter
import logging
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from sklearn.utils import resample
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from skimage.io import imread, imshow
from skimage.feature import hog
from tqdm import tqdm  # Добавлено импортирование tqdm
import cv2
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.svm import SVC
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from pprint import pprint
from time import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import joblib

smiley_faces = [
    "men1.jpg",
    "men2.jpg",
    "men3.jpg",
    "men4.jpg",
    "men5.jpg",
    "men6.jpg",
    "men7.jpg",
    "men8.jpg",
    "men9.jpg",
    "men10.jpg",
    #"men11.jpg",
    "men12.jpg",
    "men13.jpg",
    "men14.jpg",
    #"men15.jpg",
    "men16.jpg",
    "men17.jpg",
    "men18.jpg",
    "men19.jpg",
    #"men20.jpg",
    "men21.jpg",
    #"men22.jpg",
    "men23.jpg",
    #"men24.jpg",
    #"men25.jpg",
    "men26.jpg",
    "men27.jpg",
    "men28.jpg",
    "men29.jpg",
    "men30.jpg",
    "men31.jpg",
    "men32.jpg",
    "men33.jpg",
    "men34.jpg",
    "men35.jpg",
    "men36.jpg",
    "men37.jpg"

]  # Список изображений лиц
emotions = [
    "веселый",  # 1
    "веселый",  # 2
    "грустный",  # 3
    "нейтральный",  # 4
    "нейтральный",  # 5
    "грустный",  # 6
    "нейтральный",  # 7
    "нейтральный",  # 8
    "нейтральный",  # 9
    "нейтральный",  # 10
    #"грустный",  # 11
    "грустный",  # 12
    "веселый",  # 13
    "нейтральный",  # 14
    #"грустный",  # 15
    "веселый",  # 16
    "нейтральный",  # 17
    "веселый",  # 18
    "нейтральный",  # 19
    #"грустный",  # 20
    "грустный",  # 21
    #"грустный",  # 22
    "грустный",  # 23
    #"грустный",  # 24
    #"грустный",  # 25
    "нейтральный",  # 26
    "веселый",  # 27
    "веселый",  # 28
    "веселый",  # 29
    "нейтральный",  # 30
    "веселый",  # 31
    "нейтральный",  # 32
    "веселый",  # 33
    "веселый",  # 34
    "веселый",  # 35
    "веселый",  # 36
    "веселый"  # 37


]
test_image = [
    "test1.jpg",
    "test2.jpg",
    "test3.jpg",
    "test4.jpg",
    "test5.jpg",
    "test6.jpg",
    "test7.jpg"
]
good_rez = [
    "веселый",
    "нейтрально-грустный",
    "грустный",
    "веселый",
    "нейтрально-весёлый",
    "нейтрально-непонятный",
    "веселый"
]

# Список соответствующих эмоций"""
label_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
label_names = ['Злой', 'Отвращение', 'Страх', 'Счастливый', 'Грустный', 'Удивленный', 'Нейтральный']



#Предварительная обработка изображения смайлика.
def preprocess_image(image_path):
    """
    Предварительная обработка изображения смайлика.
    :param image_path: Путь к изображению смайлика.
    :return: Обработанное изображение.
    """
    # Загрузка изображения смайлика
    image = cv2.imread(image_path)
    # Изменение размера изображения до 100x100 пикселей
    image = cv2.resize(image, (100, 100))
    # Конвертация в градации серого
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Нормализация яркости и контраста
    image = cv2.equalizeHist(image)
    return image
def preprocess_image48(image_path):
    """
    Предварительная обработка изображения смайлика.
    :param image_path: Путь к изображению смайлика.
    :return: Обработанное изображение.
    """
    try:
        # Загрузка изображения смайлика
        image = cv2.imread(image_path)
        # Изменение размера изображения до 100x100 пикселей
        image = cv2.resize(image, (48, 48))
        # Конвертация в градации серого
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Нормализация яркости и контраста
        image = cv2.equalizeHist(image)

    except cv2.error as e:
        print(f"Ошибка при изменении размера изображения: {e}")
        return None
    return image

#Обнаружение области рта на изображении лица.
def detect_mouth(face_image):
    """
    Обнаружение области рта на изображении лица.
    :param face_image: Изображение лица.
    :return: Область рта на изображении.
    """
    # Загрузка каскада Хаара для обнаружения лица и рта
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
    # Обнаружение лица на изображении
    faces = face_cascade.detectMultiScale(face_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        # Определение области рта на лице
        roi_face = face_image[y:y+h, x:x+w]
        mouth = mouth_cascade.detectMultiScale(roi_face, scaleFactor=1.1, minNeighbors=5, minSize=(25, 15))
        for (mx, my, mw, mh) in mouth:
            # Возвращение области рта
            return roi_face[my:my+mh, mx:mx+mw]
    return None

#Извлечение признаков из области рта на изображении смайлика.
def extract_features(mouth_image):
    """
    Извлечение признаков из области рта на изображении смайлика.
    :param mouth_image: Область рта на изображении.
    :return: Массив признаков.
    """
    re_mouth_image=cv2.resize(mouth_image, (30, 16))
    # Настройка параметров HOG-дескриптора
    #hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    #winSize = mouth_image.shape[::-1]  # Reverse the shape to (width, height)
    hog = cv2.HOGDescriptor((16,16), (16, 16), (8, 8), (8, 8), 9)
    features = hog.compute(re_mouth_image)

    return features

#Поиск самого частого значения.
def most_frequent(lst):
    counter = Counter(lst)
    max_count = max(counter.values())
    return [item for item, count in counter.items() if count == max_count]

# Функция для предварительной обработки изображений с использованием HOG (model1)
def preprocess_with_hog(images):
    # Вычисление признаков HOG и визуализация HOG-изображений для каждого изображения
    hog_features = []
    hog_images = []
    for image in tqdm(images):
        features, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                                 cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys', visualize=True)
        hog_features.append(features)
        hog_images.append(hog_image)
    hog_features = np.array(hog_features)
    return hog_features, hog_images

#Проверка на существования файла по пути.
def check_jpg_file(file_path):
    if os.path.isfile(file_path) and file_path.lower().endswith('.jpg'):
        return True
    else:
        return False

#swich на соответствующий ответ
def get_message(emotion):
    if emotion == "Злой":
        return "Что тебя разозлило?","Я бы тоже разозлился, но у меня же нет эмоций.."
    elif emotion == "Отвращение":
        return "Что тебе так не понравилось?","Ой я тоже такое не люблю!"
    elif emotion == "Страх":
        return "Что тебя напугало?","Должен признаться, я тоже кое-чего боюсь."
    elif emotion == "Счастливый":
        return "Ого! Вот это улыбка, Вам бы на обложке журналов сниматься! Что так развеселило?","Позитив это очень хорошо!"
    elif emotion == "Грустный":
        return "Почему ты грустный?","Всё наладится!"
    elif emotion == "Удивленный":
        return "Что тебя так удивило?","Вот это да!"
    elif emotion == "Нейтральный":
        return "Как ты сегодня себя чувствуешь?","Покушай что-нибудь вкусное, кстати сахар повышает эндарфин!"
    else:
        return "Неизвестная эмоция","Даже не знаю что на это ответить.."

def predict(testImage,model):
    public_X_test = [preprocess_image48(image_path) for image_path in testImage]
    public_fds, _ = preprocess_with_hog(public_X_test)
    # Выполнение предсказаний на тестовых данных
    public_y_pred = model.predict(public_fds)
    public_y_emotions = [label_names[i] for i in public_y_pred]
    return public_y_emotions