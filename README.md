# X5-Hackatone


Обучение модели реализуется в Jupyter notebook в файле train_model.ipynb

В качестве альтернативы можно скачать модель отсюда: https://drive.google.com/drive/folders/1rrzQABDhPPhzNBOkdOmTxGHeHJzrnf1L?usp=sharing

и поместить в папку models/FacebookAI

Запуск API сервиса
```
docker build -t app-api:latest .

docker run --gpus all -p 500:5000 app-api:latest
```
---
