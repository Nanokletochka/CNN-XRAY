from ultralytics import YOLO
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Загружаем модель YOLO для классификации
model = YOLO("../../models/yolo weights/yolo11s-cls.pt")

# Загружаем данные
transform = transforms.Compose([
    transforms.ToTensor(),  # Преобразуем изображение в тензор
])

# Используем ImageFolder для загрузки данных
dataset = ImageFolder("../../dataset/test", transform=transform)

# Смотрим соответствие классов и меток
class_to_idx = dataset.class_to_idx
print("Сопоставление классов и меток:", class_to_idx)

# Генератор батчей
batch_generator = DataLoader(
    dataset,
    batch_size=1,  # Обрабатываем по одному изображению
    shuffle=False,
    drop_last=False
)

# Списки для хранения вероятностей и истинных меток
y_true_all = []  # Истинные метки (0 или 1)
y_prob_all = []  # Вероятности для класса PNEUMONIA (метка 1)

# Валидация модели
model.eval()

# Собираем ответы модели
for images, labels in tqdm(batch_generator, desc="Обработка изображений"):
    # Преобразуем тензор в формат HWC и uint8
    image_np = (images[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # Получаем предсказания для изображения
    results = model.predict(image_np)
    
    # Извлекаем вероятности классов
    probs = results[0].probs  # Объект Probs
    pneumonia_prob = probs.data[1].item()  # Вероятность для класса PNEUMONIA (метка 1)
    y_prob_all.append(pneumonia_prob)
    
    # Сохраняем истинную метку
    y_true_all.append(labels.item())

# Преобразуем списки в массивы NumPy
y_true_all = np.array(y_true_all)
y_prob_all = np.array(y_prob_all)

# Вычисляем ROC-кривую
fpr, tpr, thresholds = roc_curve(y_true_all, y_prob_all)

# Вычисляем AUC
roc_auc = roc_auc_score(y_true_all, y_prob_all)

# Построение ROC-кривой
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Случайный классификатор
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for YOLO Classification Model')
plt.legend(loc="lower right")
plt.show()

# Находим оптимальный порог (максимизация Youden's Index)
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]

print(f"Оптимальный порог: {optimal_threshold:.4f}")

# Применяем оптимальный порог для предсказаний
y_pred_all = (y_prob_all >= optimal_threshold).astype(int)

# Считаем метрики
accuracy = accuracy_score(y_true_all, y_pred_all)
precision = precision_score(y_true_all, y_pred_all)
recall = recall_score(y_true_all, y_pred_all)
f1 = f1_score(y_true_all, y_pred_all)
conf_matrix = confusion_matrix(y_true_all, y_pred_all)

# Выводим метрики
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

"""
Оптимальный порог: 0.7280
Accuracy: 0.9840
Precision: 0.9822
Recall: 0.9923
F1-score: 0.9872
Confusion Matrix:
[[227   7]
 [  3 387]]
"""
