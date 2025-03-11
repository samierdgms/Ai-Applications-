from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os

app = Flask(__name__, template_folder="templates")

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

active_models = []
tesseract_ai = huggingface_ai = whisper_ai = resnet_ai = None

ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
ALLOWED_TEXT_EXTENSIONS = {'txt'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def home():
    active_models = [
        ('Hugging Face AI', huggingface_ai),
        ('Tesseract AI', tesseract_ai),
        ('Whisper AI', whisper_ai),
        ('ResNet50 AI', resnet_ai)
    ]
    return render_template('index.html', active_models=[name for name, model in active_models if model])

@app.route('/active_models')
def get_active_models():
    return {'active_models': [name for name, model in [
        ('Hugging Face AI', huggingface_ai),
        ('Tesseract AI', tesseract_ai),
        ('Whisper AI', whisper_ai),
        ('ResNet50 AI', resnet_ai)
    ] if model]}

@app.route('/resnet50', methods=['GET', 'POST'])
def resnet50():
    global resnet_ai
    if resnet_ai is None:
        from ai_services.Resnet50_AI import ResNet50_AI
        resnet_ai = ResNet50_AI()

    analysis_result = None
    if request.method == 'POST' and (file := request.files.get('file')):
        if allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            analysis_result = resnet_ai.analyze_image(filepath)
            with open(os.path.join(app.config['OUTPUT_FOLDER'], filename.rsplit('.', 1)[0] + '_resnet50_analysis.txt'), 'w', encoding="utf-8") as f:
                f.write(analysis_result)

    return render_template('resnet50.html', analysis_result=analysis_result)

@app.route('/whisper', methods=['GET', 'POST'])
def whisper():
    global whisper_ai
    if whisper_ai is None:
        from ai_services.WhisperAI import WhisperAI
        whisper_ai = WhisperAI()

    transcription, transcription_file = None, None
    if request.method == 'POST' and (file := request.files.get('file')):
        if allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            transcription = whisper_ai.transcribe(filepath)
            output_filename = filename.rsplit('.', 1)[0] + '.txt'
            with open(os.path.join(app.config['OUTPUT_FOLDER'], output_filename), 'w', encoding="utf-8") as f:
                f.write(transcription)

            transcription_file = output_filename

    return render_template('whisper.html', transcription=transcription, transcription_file=transcription_file)

@app.route('/translator', methods=['GET', 'POST'])
def translator():
    global huggingface_ai
    if huggingface_ai is None:
        from ai_services.HuggingFace_AI import HuggingFace_AI
        huggingface_ai = HuggingFace_AI()

    translated_text, translation_file = None, None
    if request.method == 'POST':
        text = request.form.get('text')
        file = request.files.get('file')

        if file and allowed_file(file.filename, ALLOWED_TEXT_EXTENSIONS):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

        if text:
            translated_text = huggingface_ai.translate(text, request.form.get('source_language', 'tur_Latn'), request.form.get('target_language', 'eng_Latn'))
            translation_file = "translated_" + (file.filename if file else "translation.txt")
            with open(os.path.join(app.config['OUTPUT_FOLDER'], translation_file), "w", encoding="utf-8") as f:
                f.write(translated_text)

    return render_template('translator.html', translated_text=translated_text, translation_file=translation_file)

@app.route('/tesseract', methods=['GET', 'POST'])
def tesseract():
    global tesseract_ai
    if tesseract_ai is None:
        from ai_services.Tesseract_AI import Tesseract_AI
        tesseract_ai = Tesseract_AI(lang="eng")

    extracted_text, extracted_file = None, None
    if request.method == 'POST' and (file := request.files.get('file')):
        if allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            extracted_text = tesseract_ai.extract_text(filepath)
            extracted_file = filename.rsplit('.', 1)[0] + '_ocr.txt'
            with open(os.path.join(app.config['OUTPUT_FOLDER'], extracted_file), 'w', encoding="utf-8") as f:
                f.write(extracted_text)

    return render_template('tesseract.html', extracted_text=extracted_text, extracted_file=extracted_file)

@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

@app.route('/reset_models', methods=['POST'])
def reset_models():
    global whisper_ai, resnet_ai, huggingface_ai, tesseract_ai
    whisper_ai = resnet_ai = huggingface_ai = tesseract_ai = None
    return '', 204

@app.route('/reset_<model>', methods=['POST'])
def reset_model(model):
    global whisper_ai, resnet_ai, huggingface_ai, tesseract_ai
    if model == 'whisper':
        whisper_ai = None
    elif model == 'resnet50':
        resnet_ai = None
    elif model == 'huggingface':
        huggingface_ai = None
    elif model == 'tesseract':
        tesseract_ai = None
    return '', 204

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    app.run(debug=True)
