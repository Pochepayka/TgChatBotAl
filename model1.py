from defs import *

# Создание функции для получения массивов изображений из строк пикселей
def array_conversions(df):
    # Создание массивов изображений
    img_arrays = df[' pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape(48, 48))
    img_arrays = np.stack(img_arrays)
    # Создание массива меток
    label_array = df['emotion'].to_numpy()
    return img_arrays, label_array

# Создание функции для поиска изображений с только 1 уникальным значением пикселя
# Если только 1 уникальное значение пикселя, то изображение состоит из одного цвета
def same_pixel_value(row):
    return np.unique(row).size == 1

# retrieving FER input data
icml_face_data = pd.read_csv("icml_face_data.csv")

# Создание переменной train
train = icml_face_data[icml_face_data[' Usage'] == "Training"]

# Отображение информации о train
train.info()

# Разделение строк пиксельных значений и преобразование в массив
pixels = train[' pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' '))

# Получение индексов строк
same_pixel_mask = pixels.apply(same_pixel_value)
same_pixel_true_indices = same_pixel_mask[same_pixel_mask==True].index
num_of_images = len(same_pixel_true_indices)

# Удаление изображений с только одним цветом
train = train.drop(same_pixel_true_indices)

# Вычисление распределения целевых классов
td = train["emotion"].value_counts().sort_index()

# Вычисление весов классов
unique_classes = np.unique(train['emotion'])
class_counts = train['emotion'].to_list()
class_weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=class_counts)
class_weights_dict = dict(zip(unique_classes, class_weights))

# Печать весов классов
"""print("Словарь весов классов:")
for key, value in class_weights_dict.items():
    print(f"Класс {key}: {value}")"""

# Преобразование строк пикселей в массивы изображений
train_images, train_labels = array_conversions(train)

# Визуализация 25 случайных изображений из train_images
"""plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap='gray')  # Попробуйте 'viridis' или 'jet' для цветовой карты
    plt.xlabel(label_names[train_labels[i]])

# Настройка расстояния между подграфиками для лучшей визуализации
plt.tight_layout()
plt.show()
"""

# Определение классов меньшинства
minority_classes = [0, 1, 2, 4, 5, 6]

# Разделение изображений по классам
class_images = {emotion: [] for emotion in minority_classes}
for i, label in enumerate(train_labels):
    if label in minority_classes:
        class_images[label].append(train_images[i])

# Параметры аугментации данных
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Аугментация изображений и добавление к оригинальным train_images
augmented_images = []
augmented_labels = []
for emotion, images in class_images.items():
    if len(images) > 0:
        for img in images:
            img = np.expand_dims(img, axis=0)  # Добавление размерности пакета
            img = np.expand_dims(img, axis=-1)  # Добавление размерности каналов (оттенки серого в 3 канала)
            augment_iter = datagen.flow(img, batch_size=1)
            aug_img = next(augment_iter)[0]
            if emotion == 1:  # Проверка, является ли эмоция "Disgust" (предполагая, что 1 - это метка для "Disgust")
                # Аугментация класса "Disgust" 10 раз
                for _ in range(10):
                    augmented_images.append(aug_img)
                    augmented_labels.append(emotion)  # Обновление меток соответствующим образом
            else:
                # Для других классов меньшинства аугментация только один раз
                augmented_images.append(aug_img)
                augmented_labels.append(emotion)  # Обновление меток соответствующим образом

# Преобразование augmented_images и augmented_labels в массивы NumPy
augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)
"""# Проверка формы аугментированных изображений и меток
print("Форма аугментированных изображений:", augmented_images.shape)
print("Форма аугментированных меток:", augmented_labels.shape)"""

# Удаление лишней размерности из аугментированных изображений
augmented_images = np.squeeze(augmented_images, axis=-1)
"""# Проверка формы аугментированных изображений и меток
print("Форма изображений train:", train_images.shape)
print("Форма меток train:", train_labels.shape)"""

# Объединение аугментированных изображений с оригинальными
train_images_augmented = np.concatenate((train_images, augmented_images), axis=0)
train_labels_augmented = np.concatenate((train_labels, augmented_labels), axis=0)
"""# Проверка формы аугментированных данных train
print("Форма аугментированных изображений train:", train_images_augmented.shape)
print("Форма аугментированных меток train:", train_labels_augmented.shape)
"""

# Вычисление весов классов для аугментированных данных
unique_classes_augmented = np.unique(train_labels_augmented)
class_counts_augmented = train_labels_augmented.tolist()
class_weights_augmented = compute_class_weight(class_weight="balanced", classes=unique_classes_augmented, y=class_counts_augmented)
class_weights_dict_augmented = dict(zip(unique_classes_augmented, class_weights_augmented))
"""# Печать весов классов для аугментированных данных
print("Class Weights Dictionary for Augmented Data:")
for key, value in class_weights_dict_augmented.items():
    print(f"Class {key}: {value}")"""

# Получение нового распределения целевых классов
unique_classes, class_counts = np.unique(train_labels_augmented, return_counts=True)
"""plt.bar(unique_classes, class_counts)
plt.title('New Class Distribution')
plt.xlabel('Emotion')
plt.ylabel('Counts')
plt.xticks(ticks=range(len(label_names)), labels=label_names, rotation=0)
plt.show()
"""

# Печать количества образцов для каждого класса в аугментированных данных
label_counts = np.bincount(train_labels_augmented)
"""print("Class Counts in Augmented Data:")
for label, count in enumerate(label_counts):
    print(f"Class {label}: {count} samples")"""

# Применение предварительной обработки с HOG к аугментированным изображениям
hog_features, hog_images = preprocess_with_hog(train_images_augmented)
"""# Получение HOG-изображения для первого изображения в train_images
fd, hog_image = hog(train_images[0], visualize=True)

# Визуализация оригинального изображения и соответствующего HOG-изображения
plt.figure(figsize=(5, 5))
plt.subplot(1, 2, 1)
plt.imshow(train_images[0], cmap=plt.cm.gray)
plt.title('Train Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(hog_images[0], cmap=plt.cm.gray)
plt.title('HOG Image')
plt.axis('off')
plt.show()"""

# Ускорение обучения SVC
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.svm import SVC

# Обучение модели
model = SVC(class_weight=class_weights_dict, C=0.1, degree=6, kernel='poly')
model.fit(hog_features, train_labels_augmented)

# Сохранение обученной модели
joblib.dump(model, 'trained_model.pkl')

"""# Предварительная обработка тестовых данных
public_test = icml_face_data[icml_face_data[' Usage'] == "PublicTest"]
public_X_test, public_y_test = array_conversions(public_test)
public_fds, _ = preprocess_with_hog(public_X_test)

# Выполнение предсказаний на публичных тестовых данных
public_y_pred = model.predict(public_fds)

# Создание и визуализация нормализованной матрицы confusion для публичных тестовых данных
public_cm = confusion_matrix(public_y_test, public_y_pred, normalize='true')
#plt.figure(figsize=(8, 6))
sns.heatmap(public_cm, cmap='Blues', annot=True, xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Values')
plt.ylabel('True Values')
plt.title('Normalized Confusion Matrix')
plt.show()

# Создание отчета о классификации для публичных тестовых данных
public_cr = classification_report(public_y_test, public_y_pred, target_names=label_names)
print(public_cr)

# Предварительная обработка частных тестовых данных
private_test = icml_face_data[icml_face_data[' Usage'] == "PrivateTest"]
private_X_test, private_y_test = array_conversions(private_test)
private_fds, _ = preprocess_with_hog(private_X_test)

# Выполнение предсказаний на частных тестовых данных
private_y_pred = model.predict(private_fds)

# Создание и визуализация нормализованной матрицы confusion для частных тестовых данных
private_cm = confusion_matrix(private_y_test, private_y_pred, normalize='true')
#plt.figure(figsize=(8, 6))
sns.heatmap(private_cm, cmap='Blues', annot=True, xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Values')
plt.ylabel('True Values')
plt.title('Normalized Confusion Matrix')
plt.show()

# Создание отчета о классификации для частных тестовых данных
private_cr = classification_report(private_y_test, private_y_pred, target_names=label_names)
print(private_cr)"""
