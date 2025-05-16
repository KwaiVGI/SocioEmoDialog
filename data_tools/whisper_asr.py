"""
Convert all audio files in a folder into text using the Whisper model.

1. Read all audio files in the specified folder (format: .wav)
2. Perform automatic speech recognition (ASR):
    2.1 Split each audio file into non-silent segments
    2.2 Use the Whisper model to transcribe each segment into text
3. Save the transcribed text as a .json file, with the same name as the audio file

Q: Why split the audio files first?
A: The Whisper model performs better on short audio segments than on long ones, especially when using prompts.

Output format:
[
    {
        "idx": 0,
        "start": "00:00:13.497",
        "end": "00:00:17.479",
        "asr": "This is the first text segment",
        "asr_flag": true
    },
]
"""

WHISPER_MODEL = 'large-v3'
# WHISPER_MODEL = 'tiny'

import os
import sys
import argparse

from tqdm import tqdm

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

import whisper
import json

import re

import warnings
warnings.filterwarnings("ignore")


def nonsilent_ranges_detect(audio_path, temp_dir="temp_segments"):
    silence_threshold = -80     # Silence threshold (in dB)
    min_silence_length = 1000   # Minimum silence length (in milliseconds)

    # Load audio
    audio = AudioSegment.from_file(audio_path)

    # Detect non-silent segments
    nonsilent_ranges = detect_nonsilent(audio, silence_thresh=silence_threshold, min_silence_len=min_silence_length)

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    else:
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))

    for idx, (start, end) in enumerate(nonsilent_ranges):
        segment = audio[start:end]
        segment.export(f"{temp_dir}/temp_segment_{idx}.wav", format="wav")

    return nonsilent_ranges


def contains_other_language(s: str) -> bool:
    """
    Determine whether a string contains characters other than Chinese, English letters, digits, punctuation, and spaces.

    Details:
    - Chinese characters are defined in the Unicode range: \u4e00 - \u9fff
    - English letters include: A-Z and a-z
    - Digits include half-width 0-9 and full-width digits (\uff10-\uff19)
    - Punctuation includes common Chinese and English punctuation marks
    - Spaces are considered valid characters

    Return:
    Returns True if any character outside Chinese, English, digits, punctuation, or spaces is found; 
    otherwise, returns False.
    """
    # Allowed characters: Chinese, English letters, digits (half-width and full-width), 
    # common punctuation (Chinese and English), and spaces
    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~、。？！，；：“”‘’（）【】—…《》﹗﹖﹔﹐﹑﹚"
    pattern = re.compile(r'[^A-Za-z\u4e00-\u9fff0-9\uff10-\uff19\s' + re.escape(punctuation) + ']')
    return bool(pattern.search(s))


def asr_result_check(result):
    if "点赞" in result and "订阅" in result:           # Whisper hallucination
        return False
    if "中文普通话" in result:                         # Whisper hallucination
        return False
    if len(result) < 2:                                 # Too short
        return False
    if contains_other_language(result):                 # Other language
        return False
    return True
    

def asr_process(vp, wav_seg_dir):
    # Determine the log file name based on the name of the folder
    dir_path = os.path.dirname(wav_seg_dir)
    if 'left' in wav_seg_dir:
        log_file_name = os.path.basename(dir_path) + '_left_speaker_diarization.log'
    else:
        log_file_name = os.path.basename(dir_path) + '_right_speaker_diarization.log'
    log_file_path = os.path.join(dir_path, log_file_name)

    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    pattern = r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-\s*(\d{2}:\d{2}:\d{2}\.\d{3}).*\[Saved as (.*?)\]"
    segments = []
    idx = 0

    for line in lines:
        match = re.match(pattern, line)
        if match:
            start_time = match.group(1)
            end_time = match.group(2)
            wav_file = match.group(3)
            audio_path = os.path.join(wav_seg_dir, wav_file)
            result = vp.whisper_model.transcribe(
                audio_path, 
                hallucination_silence_threshold=2.0, 
                word_timestamps=True,
                initial_prompt="以下是中文普通话的句子，是一段聊天内容，输出标点。"
            )
        
            check_result = asr_result_check(result["text"])
            
            segments.append(
                {
                    "idx": idx,
                    "start": start_time,
                    "end": end_time,
                    "asr": result["text"],
                    "asr_flag": check_result,
                }
            )
            idx += 1

    save_path = log_file_path.replace('.log', '_asr.json')
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    print(f'Load whisper model {WHISPER_MODEL} ...')
    model = whisper.load_model(WHISPER_MODEL)
    print('Load whisper model done.')
    # test
    asr_process('xxx_left_speaker_diarization.log')