import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

class ResNet50_AI:
    def __init__(self):
        # Modeli yükle (önceden eğitilmiş ResNet50)
        self.model = ResNet50(weights='imagenet')

    def analyze_image(self, image_path):
        # Görüntüyü yükle
        image = cv2.imread(image_path)

        # Görüntüyü uygun boyuta getir (ResNet50 için 224x224)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV BGR formatında açar, RGB'ye çevirelim

        # Modelin anlayabileceği forma sok (Önişleme)
        image = np.expand_dims(image, axis=0)  # (224, 224, 3) → (1, 224, 224, 3)
        image = preprocess_input(image)

        # Model ile tahmin yap
        predictions = self.model.predict(image)

        # En olası 3 sınıfı getir
        decoded_predictions = decode_predictions(predictions, top=5)[0]

        # Tahmin sonuçlarını döndür
        result = ""
        for i, (id, label, score) in enumerate(decoded_predictions):
            result += f"{i + 1}. {label}: %{score * 100:.2f} olasılık\n"

        return result
