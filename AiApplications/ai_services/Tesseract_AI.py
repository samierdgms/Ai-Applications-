
import pytesseract
from PIL import Image

# Windows için Tesseract'ın yolu (gerekirse değiştir)
pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract\tesseract.exe"


class Tesseract_AI:
    def __init__(self, lang="eng"):
        self.lang = lang  # Varsayılan dil İngilizce

    def extract_text(self, image_path):
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang=self.lang)
            return text.strip()
        except Exception as e:
            return f"Error: {str(e)}"
