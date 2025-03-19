from train import CNN

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision.datasets import ImageFolder
import torch.utils.data as data
import torch.nn.functional as F


# Загружаем модель
model = CNN()

# Загружаем веса
state_dict = torch.load("../models/train0/weights_4.tar")
model.load_state_dict(state_dict)

# Загружаем данные
dataset = ImageFolder("../dataset/test", transform=model.transform())

# Смотрим сопоставление классов и меток
class_to_idx = dataset.class_to_idx
print("Сопоставление классов и меток:", class_to_idx)

# Находим имя класса для метки 1
label = 1
class_name = next((name for name, idx in class_to_idx.items() if idx == label), None)
print(f"Метка {label} соответствует классу: {class_name}")

# Генератор батчей
batch_generator = data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    drop_last=False
)

# Предсказанные и истинные метки
y_true_all = []
y_prob_all = []

model.eval()

# Собираем ответы модели
with torch.no_grad():
    for i, (x_test, y_test) in enumerate(batch_generator):
        print(f"Обработка батча {i+1}/{len(batch_generator)}")
        
        # Предсказание модели (логиты)
        logits = model(x_test)
        
        # Применяем софтмакс для получения вероятностей
        probs = F.softmax(logits, dim=1)
        
        # Для ROC-кривой нам нужны вероятности положительного класса (обычно класс 1)
        y_prob_all.append(probs[:, 1].cpu().numpy())  # Вероятности для класса 1
        y_true_all.append(y_test.cpu().numpy())

# Преобразуем списки в одномерные массивы
y_true_all = np.concatenate(y_true_all)
y_prob_all = np.concatenate(y_prob_all)

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
plt.title('Receiver Operating Characteristic (ROC) Curve')
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
Оптимальный порог: 0.4263
Accuracy: 0.8413
Precision: 0.8557
Recall: 0.8974
F1-score: 0.8761
Confusion Matrix:
[[175  59]
 [ 40 350]]
"""
