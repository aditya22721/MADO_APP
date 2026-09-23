"""
Validation Texts for Voice_Finetuning Evaluation.

Provides a fixed, balanced, and diverse set of test sentences designed
to evaluate speech synthesis across multiple linguistic domains:
    - Normal conversational speech
    - Short sentences & acknowledgments
    - Long structured sentences
    - Questions with rising intonation
    - Numbers and measurements
    - Emergency / alert notifications
    - CARE assistant style responses
"""

from typing import Dict, List

DEFAULT_VALIDATION_TEXTS: List[Dict[str, str]] = [
    {
        "id": "val_conv_01",
        "category": "conversational",
        "text": "Hello! I am here to help you throughout your daily routine."
    },
    {
        "id": "val_short_02",
        "category": "short_response",
        "text": "Everything is in order."
    },
    {
        "id": "val_long_03",
        "category": "long_sentence",
        "text": "Please remember to take your afternoon medication with a full glass of water right after lunch."
    },
    {
        "id": "val_question_04",
        "category": "question",
        "text": "Would you like me to adjust the room temperature or turn on the reading light for you?"
    },
    {
        "id": "val_numbers_05",
        "category": "numbers_and_measures",
        "text": "Your heart rate is 72 beats per minute, and your blood pressure was recorded at 120 over 80."
    },
    {
        "id": "val_emergency_06",
        "category": "emergency_alert",
        "text": "Please remain seated while I immediately notify your caregiver."
    },
    {
        "id": "val_care_07",
        "category": "care_assistant",
        "text": "I hope you had a restful afternoon sleep, and I am glad to see you feeling better today."
    },
    {
        "id": "val_phonetic_08",
        "category": "phonetic_variety",
        "text": "The quick brown fox jumps smoothly over lazy yellow dogs near the quiet brook."
    }
]


def get_validation_texts(custom_texts: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
    """Return the active list of validation texts."""
    if custom_texts is not None and len(custom_texts) > 0:
        return custom_texts
    return DEFAULT_VALIDATION_TEXTS


def get_validation_sentences(custom_texts: List[Dict[str, str]] = None) -> List[str]:
    """Return plain list of sentence strings for generation."""
    texts = get_validation_texts(custom_texts)
    return [item["text"] for item in texts]
