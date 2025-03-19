from torchvision.datasets import ImageFolder
import torch.nn as nn
import torchvision.transforms.v2 as tfs
import torch
import torch.utils.data as data
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class CNN(nn.Module):
    # Кол-во параметров модели 10,357,346
    
    def __init__(self):
        super().__init__()

        # Модель
        self.model = nn.Sequential(
            # Свёрточные слои
            nn.Conv2d(1, 16, 3, padding=1, stride=2),  # 800x800 -> 400x400
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 400x400 -> 200x200

            nn.Conv2d(16, 32, 3, padding=1, stride=2),  # 200x200 -> 100x100
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 100x100 -> 50x50

            nn.Conv2d(32, 64, 3, padding=1),  # 50x50 -> 50x50
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 50x50 -> 25x25

            nn.Conv2d(64, 128, 3, padding=1),  # 25x25 -> 25x25
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 25x25 -> 12x12

            nn.Conv2d(128, 256, 3, padding=1),  # 12x12 -> 12x12
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12x12 -> 6x6

            nn.Dropout(0.5),
            nn.Flatten(),

            # Полносвязные слои
            nn.Linear(256 * 6 * 6, 1024),  # Вход: 256 * 6 * 6, выход: 1024
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),  # Вход: 1024, выход: 512
            nn.ReLU(),
            nn.Linear(512, 2)  # Выходной слой (2 класса)
        )

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def transform():
        """Применяет трансформации к тестовым изображениям."""
        
        # Трансформации для тестовых данных (без аугментации)
        test_transforms = tfs.Compose([
            tfs.Resize(800),
            tfs.CenterCrop(800),  # Обрезаем центр до 800х800
            tfs.ToImage(),  # Преобразование в тензор
            tfs.Grayscale(),  # Преобразование в градации серого
            tfs.ToDtype(torch.float32, scale=True),  # Нормализация
        ])

        return test_transforms


if __name__ == "__main__":
    # Инициализируем модель
    model = CNN()
    model.train()

    # Аугментация для обучающих данных
    train_transforms = tfs.Compose([
        tfs.Resize(800),
        tfs.CenterCrop(800),  # Обрезаем центр до 800х800
        tfs.RandomHorizontalFlip(),  # Случайное отражение по горизонтали
        tfs.RandomRotation(15),  # Случайный поворот на ±15 градусов
        tfs.ToImage(),  # Преобразование в тензор
        tfs.Grayscale(),  # Преобразование в градации серого
        tfs.ToDtype(torch.float32, scale=True),  # Нормализация
    ])

    # Загрузка данных с аугментацией
    dataset_train = ImageFolder("../../dataset/train", transform=train_transforms)
    dataset_test = ImageFolder("../../dataset/test", transform=model.transform)

    # Генератор батчей
    batch_generator_train = data.DataLoader(
        dataset_train,
        batch_size=32,
        shuffle=True,
        drop_last=False
    )

    batch_generator_test = data.DataLoader(
        dataset_test,
        batch_size=32,
        shuffle=True,
        drop_last=False
    )

    # История обучения
    train_loss_hist = []
    test_loss_hist = []

    # Оптимизатор
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)

    # Функция потерь
    loss = nn.CrossEntropyLoss()

    # Кол-во эпох обучения
    epochs = 10

    # Градиентный спуск
    for epoch in range(epochs):
        # Cредняя ошибка на трейне за эпоху
        mean_train_loss_value = 0

        with tqdm(batch_generator_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for x_train, y_train in pbar:
                # Предсказание модели
                y_pred = model(x_train)
                # Подставляем предсказание в функцию потерь
                loss_value = loss(y_pred, y_train)
                # Обнуляем градиенты
                optimizer.zero_grad()
                # Вычисляем градиент по фунции потерь в точке
                loss_value.backward()
                # Шаг градиентного спуска
                optimizer.step()

                # Обновляем описание прогресс-бара с текущим значением потерь
                pbar.set_postfix({"Train Loss": loss_value.item()})
                # Обновляет значение средней ошибки
                mean_train_loss_value += loss_value.item()

            # Добавляем среднее значение ошибки за эпоху на тесте
            mean_train_loss_value = mean_train_loss_value / len(batch_generator_train)
            train_loss_hist.append(mean_train_loss_value)

            # Сохраняем модель
            st = model.state_dict()
            torch.save(st, f"../../models/custom model weights/weights_{epoch}.tar")

            # Вычисляем среднюю ошибку на тестовой выборке
            model.eval()
            mean_test_loss_value = 0

            with torch.no_grad():
                for X, y in batch_generator_test:
                    y_pred = model(X)
                    loss_value = loss(y_pred, y)
                    mean_test_loss_value += loss_value.item()

                mean_test_loss_value = mean_test_loss_value/len(batch_generator_test)
                test_loss_hist.append(mean_test_loss_value)

            model.train()

    # Показываем историю обучения
    # Построение графиков истории ошибок
    plt.figure(figsize=(10, 6))  # Задаём размер графика

    # График ошибки на трейне
    plt.plot(train_loss_hist, label="Train Loss", marker="o", linestyle="-", color="blue")

    # График ошибки на тесте
    plt.plot(test_loss_hist, label="Test Loss", marker="o", linestyle="-", color="red")

    # Добавляем подписи
    plt.title("Train and Test Loss over Epochs")  # Заголовок
    plt.xlabel("Epoch")  # Ось X: эпохи
    plt.ylabel("Loss")  # Ось Y: значение ошибки
    plt.legend()  # Легенда
    plt.grid(True)  # Сетка для удобства

    # Показываем график
    plt.show()
