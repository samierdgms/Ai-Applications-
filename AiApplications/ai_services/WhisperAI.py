from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os
import whisper

app = Flask(__name__, template_folder="templates")


# Whisper modelini yükle
class WhisperAI:
    def __init__(self):
        self.model = whisper.load_model(
            "small", device="cpu")  # Burada farklı model boyutları kullanılabilir: "base", "small", "medium", "large"

    def transcribe(self, audio_path):
        result = self.model.transcribe(audio_path)
        return result['text']


# WhisperAI sınıfını başlatıyoruz
whisper_ai = WhisperAI()

# Yükleme yapılacak klasörü belirleyin
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Yüklenebilecek dosya türlerini sınırlayın
ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'mpweg', 'mpga', 'm4a', 'wav',  'webm'}


# Dosya uzantısının izinli olup olmadığını kontrol eden fonksiyon
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/whisper', methods=['GET', 'POST'])
def whisper_page():
    transcription = None
    transcription_file = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']

        if file.filename == '':
            return 'No selected file'

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Whisper ile sesi metne çevir
            transcription = whisper_ai.transcribe(filepath)

            # Metni bir dosyaya kaydedelim
            output_filename = filename.rsplit('.', 1)[0] + '.txt'
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

            with open(output_path, 'w') as output_file:
                output_file.write(transcription)

            transcription_file = output_filename

    return render_template('whisper.html', transcription=transcription, transcription_file=transcription_file)


@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    app.run(debug=True)
