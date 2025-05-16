import openai
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils_eval.file_utils import load_json

# Set your OpenAI API key
openai.api_key = "your-api-key"


# Call OpenAI GPT to classify the emotion of a conversation
def gpt_emotion_classification(text):
    prompt = (
        "Classify the overall emotion of the following conversation into one word "
        "from this list: 'Joy', 'Love', 'Anxiety', 'Satisfaction', 'Alertness', 'Hope', "
        "'Sadness', 'Amusement', 'Pride', 'Disgust', 'Anger', 'Gratitude', 'Guilt', "
        "'Fear', 'Awe', 'Offense', 'Embarrassment', 'Contempt'. "
        "Answer using only that one word.\n\n"
        f"Conversation:\n{text}"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4"
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result = response['choices'][0]['message']['content'].strip()
        return result, {"code": 1}
    except Exception as e:
        print(f"GPT call failed: {e}")
        return None, {"code": 0}


def classify_dialogue_emotion(dialogue: str) -> str:
    """
    Classify the overall emotion of a dialogue.

    Parameters:
        dialogue (str): The full dialogue content as a single string.

    Returns:
        str: The classified emotion (e.g., 'Joy', 'Anger', etc.)
    """

    emo, status = gpt_emotion_classification(dialogue)
    while status["code"] != 1:
        print("Retrying due to GPT error...")
        time.sleep(2)
        emo, status = gpt_emotion_classification(dialogue)
    
    return emo


# sample
if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent.parent
    sample_path = project_dir / 'data/scripts/scripts_sample.json'

    dataset = load_json(sample_path)

    # Iterate over each dialogue in the dataset
    for i, dialogue in enumerate(dataset.values()):
        # Concatenate all utterances into a single text
        text = "".join([utt["trans_text"] for utt in dialogue["dialogue"]])

        # Use GPT to classify emotion
        emo, status = gpt_emotion_classification(text)
        while status["code"] != 1:
            print("Retrying due to GPT error...")
            time.sleep(1)
            emo, status = gpt_emotion_classification(text)

        print(text, emo)