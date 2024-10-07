"""
This module provides a web interface for real-time language translation
using MarianMT models and Hugging Face Transformers.
"""

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
    """
    Load the MarianMT model and tokenizer for the given language pair.
    The model is cached after being loaded once.

    Args:
        source_lang (str): Source language code.
        target_lang (str): Target language code.

    Returns:
        model, tokenizer: The MarianMT model and tokenizer for the specified language pair.
    """
    key = (source_lang, target_lang)
    if key not in models:
        model_name = LANGUAGE_MODELS.get(key)
        if model_name:
            models[key] = MarianMTModel.from_pretrained(model_name)
            tokenizers[key] = MarianTokenizer.from_pretrained(model_name)
        else:
            raise ValueError(f"No model available for language pair: "
                             f"{source_lang} to {target_lang}")
    return models[key], tokenizers[key]

def detect_language(text):
    """
    Detect the language of a given text using the langid library.

    Args:
        text (str): The text whose language needs to be detected.

    Returns:
        str: Detected language code.
    """
    lang, _ = langid.classify(text)
    return lang

@app.route('/')
def index():
    """
    Render the main page with the translation form.

    Returns:
        str: Rendered HTML page.
    """
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    """
    Handle translation requests from the client and return the translated text as JSON.

    The client sends a JSON object containing the source language, target language, and sentence
    to be translated.

    Returns:
        dict: JSON response with the translated sentence.
    """
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
        translated_sentence = model.generate(
            **tokenizer(sentence, return_tensors="pt", padding=True)
        )
        translated_sentence = tokenizer.decode(
            translated_sentence[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
            # Explicitly set clean_up_tokenization_spaces
        )

        # Return the translated sentence in a JSON response
        response = {
            'translated_sentence': translated_sentence
        }
    except ValueError as e:
        response = {'translated_sentence': f"Error: {e}"}
    except RuntimeError as e:
        response = {'translated_sentence': f"Runtime error occurred: {e}"}
    except KeyError as e:
        response = {'translated_sentence': f"Key error occurred: {e}"}
    except TypeError as e:
        response = {'translated_sentence': f"Type error occurred: {e}"}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
