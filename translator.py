"""
Module for translating sentences from English to Spanish using MarianMTModel.
"""

import warnings  # Standard library import

from transformers import MarianMTModel, MarianTokenizer  # Third-party imports
import torch

# Suppress FutureWarnings from tokenization spaces
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Data Preprocessing (Tokenizers, Data Preparation)

# Example: Manually defining some sentences
source_texts = [
    "I want to translate this sentence",
    "How are you?",
    "What is your name?",
    "This is an example."
]

# Modify target texts to include <start> and <end> tokens
target_texts = [
    "<start> Quiero traducir esta oración <end>",
    "<start> ¿Cómo estás? <end>",
    "<start> ¿Cuál es tu nombre? <end>",
    "<start> Este es un ejemplo <end>"
]

# Load Marian Tokenizer
PRETRAINED_MODEL_NAME = 'Helsinki-NLP/opus-mt-en-es'
tokenizer = MarianTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

# Tokenize the source and target texts
source_sequences = tokenizer(source_texts, return_tensors="pt", padding=True, truncation=True)
target_sequences = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True)

# Model Definition (PyTorch-based Marian model)

# Load the Marian model for translation
pretrained_model = MarianMTModel.from_pretrained(PRETRAINED_MODEL_NAME)

# If GPU is available, move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model.to(device)

def translate_pretrained(sentences):
    """
    Translates a list of sentences from English to Spanish using the MarianMTModel.

    Args:
        sentences (list): List of sentences to translate.

    Returns:
        list: Translated sentences.
    """
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
    translated_tokens = pretrained_model.generate(**inputs)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]

def run_translation():
    """
    Real-time translation function. Asks the user for input, translates the input using
    the pretrained MarianMTModel, and outputs the translated sentence.
    """
    try:
        while True:
            input_sentence = input("Enter a sentence to translate (or type 'exit' to quit): ")

            if input_sentence.lower() == "exit":
                print("Exiting...")
                break

            # Batch translate the sentence(s)
            translated_sentence = translate_pretrained([input_sentence])
            print("Translated Sentence:", translated_sentence[0])

    except KeyboardInterrupt:
        print("\nTranslation process interrupted. Exiting gracefully...")

    except ValueError as e:  # Handle specific error types
        print(f"Value error occurred: {e}")

    except RuntimeError as e:  # Handle runtime errors
        print(f"Runtime error occurred: {e}")

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"An unexpected error occurred: {e}")

# Entry point for running the translation function
if __name__ == "__main__":
    run_translation()
