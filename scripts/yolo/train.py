from ultralytics import YOLO

# Предобученная модель
model = YOLO("../../models/yolo weights/yolo11s-cls.pt")

# Абсолютный путь до корневой папки датасета
path = "D://CNN//dataset"

# Начинаем обучение
results = model.train(
    data=path,
    epochs=11
    )
