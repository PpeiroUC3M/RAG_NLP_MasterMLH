# ============================================================
# LANGUAGE MODEL VALIDATOR: Check Translation Model Support
# ============================================================

"""
Summary:
This script cross-checks which languages (among those detectable by `langdetect`)
have compatible Helsinki-NLP translation models (both to and from English).
It attempts to load these models via HuggingFace Transformers, and saves a dictionary
of supported language codes with human-readable names to a JSON file.
"""

# Import necessary libraries
import os
import pycountry
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# List of languages supported by the `langdetect` library
can_detect = [
    "af", "ar", "bg", "bn", "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "fa", "fi", "fr", "gu", "he",
    "hi", "hr", "hu", "id", "it", "ja", "kn", "ko", "lt", "lv", "mk", "ml", "mr", "ne", "nl", "no", "pa", "pl",
    "pt", "ro", "ru", "sk", "sl", "so", "sq", "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "vi",
    "zh-cn", "zh-tw"
]

# Initialize list for languages with working translation models
languages_supported = []

# Check which languages have working translation models in both directions (to/from English)
for language in can_detect:
    try:
        # Try loading tokenizer and model for both directions (lang→en and en→lang)
        AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{language}-en")
        AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-en-{language}")
        AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-{language}-en")
        AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-en-{language}")

        # If no exception, add language to supported list
        languages_supported.append(language)
    except:
        # Skip any language that causes an import error
        pass

# Create a dictionary mapping language codes to readable names
lang_dict = {}

for code in languages_supported:
    try:
        # Special handling for Chinese variants
        if code == "zh-cn":
            lang_dict[code] = "Chinese (Simplified)"
        elif code == "zh-tw":
            lang_dict[code] = "Chinese (Traditional)"
        else:
            # Use pycountry to convert ISO code to full language name
            language = pycountry.languages.get(alpha_2=code)
            lang_dict[code] = language.name if language else "Unknown"
    except:
        # Fallback if pycountry fails
        lang_dict[code] = "Unknown"

# Display supported languages in the console
print("Languages in both lists:")
for code, name in lang_dict.items():
    print(f"{code} - {name}")

# Save the resulting language dictionary as JSON
saving_path = os.path.join('data', 'processed', "language_dict.json")
with open(saving_path, "w", encoding="utf-8") as json_file:
    json.dump(lang_dict, json_file, ensure_ascii=False, indent=4)
