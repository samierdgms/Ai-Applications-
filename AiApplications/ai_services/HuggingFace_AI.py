from transformers import pipeline

class HuggingFace_AI:
    def __init__(self):
        self.translator = pipeline("translation", model="facebook/nllb-200-distilled-600M")

    def translate(self, text, src_language="tur_Latn", dest_language="eng_Latn"):
        try:
            translated = self.translator(text, src_lang=src_language, tgt_lang=dest_language)
            return translated[0]["translation_text"]
        except Exception as e:
            return f"Çeviri hatası: {str(e)}"
