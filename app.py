from flask import Flask, render_template, request, jsonify
from transformers import MarianMTModel, MarianTokenizer
import langid

app = Flask(__name__)

# Load models once at startup
LANGUAGE_MODELS = {
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de"
}

models = {}
tokenizers = {}

def load_model_once(source_lang, target_lang):
    key = (source_lang, target_lang)
    if key not in models:
        model_name = LANGUAGE_MODELS.get(key)
        if model_name:
            models[key] = MarianMTModel.from_pretrained(model_name)
            tokenizers[key] = MarianTokenizer.from_pretrained(model_name)
        else:
            raise ValueError(f"No model available for language pair: {source_lang} to {target_lang}")
    return models[key], tokenizers[key]

def detect_language(text):
    lang, _ = langid.classify(text)
    return lang

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    # Extract the JSON data from the POST request
    data = request.get_json()
    source_lang = data.get('source_lang')
    target_lang = data.get('target_lang')
    sentence = data.get('sentence')

    # Handle auto-detection of the source language
    if source_lang == 'auto':
        source_lang = detect_language(sentence)

    try:
        # Load model and tokenizer once
        model, tokenizer = load_model_once(source_lang, target_lang)
        # Perform translation
        translated_sentence = model.generate(**tokenizer(sentence, return_tensors="pt", padding=True))
        translated_sentence = tokenizer.decode(translated_sentence[0], skip_special_tokens=True)

        # Return the translated sentence in a JSON response
        response = {
            'translated_sentence': translated_sentence
        }
    except ValueError as e:
        response = {
            'translated_sentence': f"Error: {e}"
        }
    except Exception as e:
        response = {
            'translated_sentence': f"An unexpected error occurred: {e}"
        }

    # Return the response as JSON
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
