from defs import *

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


]  # Список соответствующих эмоций"""

mouth_features = []  # Список для хранения извлеченных признаков из области рта
emotions_labels = []  # Список для хранения меток эмоций

for i in range(len(smiley_faces)):
    #print(i)
    face = preprocess_image(smiley_faces[i])  # Предобработка изображения и обнаружение лица
    mouth_region = detect_mouth(face)  # Обнаружение области рта
    if mouth_region is None:
        logging.warning("Регион рта не найден!")
    else:
        #print ("рот найден")
        logging.info("Область рта найдена!")
    try:
        mouth_features.append(extract_features(mouth_region))  # Извлечение признаков из области рта
        logging.info("Область рта извлечена!")
        #print ("рот извлечён")
        emotions_labels.append(emotions[i])  # Добавление метки эмоции в список
    except:
        logging.warning("Область рта не извлечена!")

try:
    mouth_features = np.vstack(mouth_features)
    X_train, X_val, y_train, y_val = train_test_split(mouth_features, emotions_labels, test_size=0.1, random_state=random.randint(5,50) ) # Разделение набора данных на обучающий и проверочный
    logging.info("НАбор тестов скомпанован!")
    #print("выборка готова")
except:
    logging.warning("Не все массивы заполены данными! Обучение невозможно!!")

#print("X_train:",X_train)
#print("X_val:",X_val)
#print("Y_train:",y_train)
#print("Y_val:",y_val)

model = svm.SVC()  # Создание модели машинного обучения (SVM)
model.fit(X_train, y_train)  # Обучение модели на обучающем наборе данных

# Сохранение обученной модели
joblib.dump(model, 'trained_model0.pkl')

y_pred=(model.predict(np.array(X_val)))  # Предсказание эмоций на проверочном наборе данных
#accuracy = accuracy_score(y_val[0], [y_pred])  # Оценка точности модели
#print("Сравни: ",y_pred,y_val)
#print("Обучение завершено!!!")

def predict_emotion(image):
    face = preprocess_image(image)  # Предобработка нового изображения смайлика
    try:
        mouth_region = detect_mouth(face)  # Обнаружение области рта
        mouth_features = extract_features(mouth_region)  # Извлечение признаков из области рта
        mouth_features = np.vstack(mouth_features)
    except:
        logging.warning("Не найдено лицо! Результат не определен!")
        return None
    X_test = np.repeat(mouth_features, 72, axis=1)

    emotion = model.predict(X_test)
    # Предсказание эмоции с обновленными данными
    lst = emotion
    result = most_frequent(lst)
    return result  # Возвращение предсказанной эмоции


