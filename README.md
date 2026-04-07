# X5-Hackatone


Обучение модели реализуется в Jupyter notebook в файле train_model.ipynb

В качестве альтернативы можно взять модель отсюда: https://drive.google.com/drive/folders/1gTmIwtWChtqjWIsuoS6lrwmn3vdXsLzq?usp=drive_link

и поместить в папку models/FacebookAI

Запуск API сервиса:
```
docker build -t app-api:latest .

docker run --gpus all -p 500:5000 app-api:latest
```
---

Пример работы сервиса:

<img width="570" height="126" alt="image" src="https://github.com/user-attachments/assets/8930e77f-e9b7-4626-9b2e-afe76e963aa7" />
