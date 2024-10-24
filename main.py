import cv2
import pickle
import os
from PIL import Image, ImageTk
import tkinter as tk


# Класс для работы с детекцией лиц и сохранением данных
class FaceDetectorScene:
    def __init__(self):
        # Инициализация каскадов для детекции лиц
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


        self.users_data = []  # Список для хранения данных пользователей

    def process_frame(self, frame):
        # Преобразуем кадр в серый для лучшей обработки
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Обнаружение лиц
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            # Рисуем прямоугольник вокруг лица
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame

    def add_person(self, name, photo_path):
        # Добавляем пользователя в список
        person = {"name": name, "photo": photo_path}
        self.users_data.append(person)

    def save_photo(self, frame, name):
        # Сохраняем фото под именем пользователя
        if not os.path.exists("photos"):
            os.makedirs("photos")  # Создаем папку, если ее нет

        photo_path = os.path.join("photos", f"{name}.jpg")
        cv2.imwrite(photo_path, frame)

        print(f"Saving photo at: {photo_path}")  # Отладочный вывод

        return photo_path

    def save_data(self, filename="users_data.pkl"):
        # Сохранение данных пользователей
        with open(filename, 'wb') as file:
            pickle.dump(self.users_data, file)

    def load_data(self, filename="users_data.pkl"):
        # Загрузка данных пользователей
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                self.users_data = pickle.load(file)


# Класс для создания интерфейса и отображения информации о пользователях
class PersonInfoUI:
    def __init__(self, root):
        self.root = root
        self.root.title("User Information")
        self.canvas = tk.Canvas(root, width=300, height=300)
        self.canvas.pack()

    def create_person_info(self, name, photo_path):
        # Открытие и обработка изображения
        img = Image.open(photo_path)
        img = img.resize((100, 100), Image.Resampling.LANCZOS)  # Исправленная строка
        img.save(os.path.join("photos", f"{name}_resized.jpg"))

        # Остальная логика создания информации о пользователе
        # ...


def main():
    face_detector = FaceDetectorScene()

    # Загружаем ранее сохраненные данные пользователей
    face_detector.load_data()

    # Включаем камеру
    cap = cv2.VideoCapture(0)

    # Запускаем камеру и интерфейс для ввода имени пользователя
    root = tk.Tk()
    ui = PersonInfoUI(root)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Обрабатываем кадр для обнаружения лиц
        frame = face_detector.process_frame(frame)

        # Отображаем кадр
        cv2.imshow("Face Detection", frame)

        # Нажмите 's' для сохранения фото и ввода имени
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # Ввод имени через интерфейс
            name = input("Введите имя пользователя: ")

            # Сохраняем фото и добавляем пользователя
            photo_path = face_detector.save_photo(frame, name)
            face_detector.add_person(name, photo_path)

            # Показываем интерфейс с информацией
            ui.create_person_info(name, photo_path)

            # Сохраняем данные
            face_detector.save_data()

        # Нажмите 'q' для выхода
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Закрываем камеру и окна
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
