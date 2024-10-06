"""
Module for translating sentences between various language pairs using MarianMTModel.
"""

import warnings  # Standard library
from transformers import MarianMTModel, MarianTokenizer  # Third-party imports
import torch

# Suppress FutureWarnings from tokenization spaces
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Map of language pairs and their corresponding MarianMT model names
LANGUAGE_MODELS = {
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",  # English to Spanish
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",  # English to French
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de"   # English to German
}

def load_model(source_lang, target_lang):
    """
    Loads the MarianMTModel and tokenizer for the given language pair.

    Args:
        source_lang (str): The source language code (e.g., 'en').
        target_lang (str): The target language code (e.g., 'es').

    Returns:
        MarianMTModel, MarianTokenizer: The model and tokenizer for the language pair.
    """
    model_name = LANGUAGE_MODELS.get((source_lang, target_lang))

    if not model_name:
        raise ValueError(f"No model available for the language pair: {source_lang} -> {target_lang}")

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    return model, tokenizer

def translate_pretrained(sentences, model, tokenizer):
    """
    Translates a list of sentences using the MarianMTModel.

    Args:
        sentences (list): List of sentences to translate.
        model: The MarianMTModel for translation.
        tokenizer: The tokenizer corresponding to the model.

    Returns:
        list: Translated sentences.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
    translated_tokens = model.generate(**inputs)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]

def run_translation():
    """
    Real-time translation function. Asks the user for input, selects the language pair,
    translates the input using the appropriate MarianMTModel, and outputs the translated sentence.
    """
    try:
        # Ask the user to select the language pair
        source_lang = input("Enter source language (e.g., 'en' for English): ").strip()
        target_lang = input("Enter target language (e.g., 'es' for Spanish): ").strip()

        model, tokenizer = load_model(source_lang, target_lang)

        while True:
            input_sentence = input("Enter a sentence to translate (or type 'exit' to quit): ")

            if input_sentence.lower() == "exit":
                print("Exiting...")
                break

            translated_sentence = translate_pretrained([input_sentence], model, tokenizer)
            print("Translated Sentence:", translated_sentence[0])

    except KeyboardInterrupt:
        print("\nTranslation process interrupted. Exiting gracefully...")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Entry point for running the translation function
if __name__ == "__main__":
    run_translation()
