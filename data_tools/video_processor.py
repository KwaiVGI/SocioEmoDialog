import torch
from pyannote.audio import Pipeline

import subprocess
import os, re
from tqdm import tqdm

from speak_diarization import diarization_process
from whisper_asr import asr_process

import whisper
WHISPER_MODEL = 'large-v3'


class VideoProcessor:
    def __init__(self):
        self.video_path = None
        self.pipeline = self.init_pyannote_pipeline()
        print(f'Load whisper model {WHISPER_MODEL} ...')
        self.whisper_model = whisper.load_model(WHISPER_MODEL)
        print('Load whisper model done.')

    def init_path(self, video_path):
        self.video_path = video_path
        base_dir = os.path.splitext(video_path)[0]
        os.makedirs(base_dir, exist_ok=True)

        # wav path of mp4
        self.wav_left_path = os.path.join(base_dir, os.path.basename(base_dir) + "_left.wav")
        self.wav_right_path = os.path.join(base_dir, os.path.basename(base_dir) + "_right.wav")
        self.wav_left_seg_dir = os.path.join(base_dir, "wav_left_segments")
        self.wav_right_seg_dir = os.path.join(base_dir, "wav_right_segments")
        self.video_seg_dir = os.path.join(base_dir, "video_segments")

    def init_pyannote_pipeline(self):
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=True,
        )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        return pipeline

    def cut_video(self):
        log_save_path = self.wav_left_path.replace(".wav", "_speaker_diarization.log")
        with open(log_save_path, 'r') as f:
            lines = f.readlines()
        counter = 0
        for line in lines:
            match = re.match(r"(\d{2}:\d{2}:\d{2}.\d{3})\s*-\s*(\d{2}:\d{2}:\d{2}.\d{3})", line)
            if match:
                start_time = match.group(1)
                end_time = match.group(2)
                output_path = os.path.join(self.video_seg_dir, "{:06d}.mp4".format(counter))
                cut_video_ffmpeg(self.video_path, output_path, start_time, end_time)
                counter += 1

    def process(self):
        # have to init path before processing!
        assert self.video_path != None

        extract_audio_ffmpeg(self.video_path, self.wav_left_path, 0)
        extract_audio_ffmpeg(self.video_path, self.wav_right_path, 1)

        diarization_process(self, self.wav_left_path, self.wav_left_seg_dir)
        diarization_process(self, self.wav_right_path, self.wav_right_seg_dir)
        self.cut_video()
        
        asr_process(self, self.wav_left_seg_dir)
        asr_process(self, self.wav_right_seg_dir)


def extract_audio_ffmpeg(video_path, audio_path, channel=0):
    if os.path.exists(audio_path):
        return
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    map_channel = f"0.1.{channel}"
    cmd = [
        "ffmpeg", 
        "-y", 
        "-i", video_path,
        "-vn",
        "-map_channel", map_channel,
        audio_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def cut_video_ffmpeg(input_path, output_path, start_time, end_time):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_time),
        "-to", str(end_time),
        "-i", input_path,
        "-c", "copy",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    vp = VideoProcessor()
    paths = ['./video/test.mp4']
    for path in tqdm(paths):
        vp.init_path(path)
        vp.process()