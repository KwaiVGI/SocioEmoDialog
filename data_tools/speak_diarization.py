import torch
from pyannote.audio import Pipeline

import librosa
import soundfile as sf
import numpy as np

import time
import os
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def sec_to_hhmmss(sec):
    """
    Convert time in seconds to the hh:mm:ss.ms format
    """
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def merge_segments(segments, gap_threshold=0.1):
    """
    Merge adjacent and continuous segments of the same speaker if the gap is less than gap_threshold seconds
    """
    segments = sorted(segments, key=lambda x: x[0])
    merged = []
    for seg in segments:
        start, end, speaker = seg
        if merged and speaker == merged[-1][2] and start - merged[-1][1] < gap_threshold:
            # If the segment belongs to the same speaker as the previous one and the gap is short, merge them
            merged[-1] = (merged[-1][0], end, speaker)
        else:
            merged.append(seg)
    return merged

def compute_speaker_energy(audio, sr, segments):
    """
    Compute the average energy for each speaker based on the RMS energy of each segment, weighted by its duration
    """
    speaker_energy = {}
    speaker_duration = {}
    for seg in segments:
        start, end, speaker = seg
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment_audio = audio[start_sample:end_sample]
        if len(segment_audio) == 0:
            continue
        # Compute RMS
        rms = np.sqrt(np.mean(segment_audio**2))
        duration = end - start
        speaker_energy[speaker] = speaker_energy.get(speaker, 0) + rms * duration
        speaker_duration[speaker] = speaker_duration.get(speaker, 0) + duration
    # Compute weighted average energy
    avg_energy = {spk: speaker_energy[spk] / speaker_duration[spk] for spk in speaker_energy}
    return avg_energy

def merge_similar_speakers(avg_energy, threshold=0.1):
    """
    Cluster speakers with similar energy levels.
    @Parameters:
        avg_energy: A dictionary where each key is the original speaker label and the value is the corresponding average energy.
        threshold: If the absolute difference in average energy between two speakers is less than this threshold, they are considered similar.
    @Returns:
        clusters: A list of clusters, where each cluster is a list of (speaker, energy) tuples.
    """
    # Sorted by avg energy
    sorted_speakers = sorted(avg_energy.items(), key=lambda x: x[1])
    clusters = []
    for spk, energy in sorted_speakers:
        if not clusters:
            clusters.append([(spk, energy)])
        else:
            # Compare with the last speaker in the current cluster using absolute energy difference
            _, last_energy = clusters[-1][-1]
            if abs(energy - last_energy) < threshold:
                clusters[-1].append((spk, energy))
            else:
                clusters.append([(spk, energy)])
    return clusters

def assign_merged_labels(avg_energy, merge_threshold=0.1):
    """
    Remap the original speaker labels based on the energy-based clustering results.
    Merge speakers with similar energy levels, and assign new labels according to the average energy in descending order:
    the speaker with the highest energy is assigned as speaker 0, the next as speaker 1, and so on.
    """
    clusters = merge_similar_speakers(avg_energy, threshold=merge_threshold)
    # Compute the average energy for each cluster
    cluster_info = []
    for cluster in clusters:
        speakers = [spk for spk, _ in cluster]
        # Average energy of the cluster
        cluster_avg = np.mean([energy for _, energy in cluster])
        cluster_info.append((speakers, cluster_avg))
    
    # Sort clusters by average energy in descending order
    cluster_info = sorted(cluster_info, key=lambda x: x[1], reverse=True)
    
    speaker_mapping = {}
    for new_label, (speakers, _) in enumerate(cluster_info):
        for spk in speakers:
            # Assign new label in the format "speaker X"
            speaker_mapping[spk] = f"speaker {new_label}"
    return speaker_mapping

def diarization_process(vp, audio_file, wav_seg_dir):
    if torch.cuda.is_available():
        vp.pipeline.to(torch.device("cuda"))
    mute_save_path = audio_file.replace(".wav", "_mute.wav")
    log_save_path = audio_file.replace(".wav", "_speaker_diarization.log")
    seg_save_path = wav_seg_dir
    if not os.path.exists(seg_save_path):
        os.makedirs(seg_save_path)
    else:
        for file in os.listdir(seg_save_path):
            os.remove(os.path.join(seg_save_path, file))

    diarization = vp.pipeline(audio_file)
    
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))
    
    # Perform initial merging of adjacent and continuous segments 
    # (gap_threshold can be adjusted as needed; set to 1 second here)
    merged_segments = merge_segments(segments, gap_threshold=1)
    
    audio, sr = librosa.load(audio_file, sr=None)
    # Compute the average RMS energy for each original speaker
    avg_energy = compute_speaker_energy(audio, sr, merged_segments)
    # Merge speakers with similar energy levels (absolute difference < 0.1)
    speaker_mapping = assign_merged_labels(avg_energy, merge_threshold=0.02)
    # Update each segment's speaker label based on the new mapping
    updated_segments = [(start, end, speaker_mapping.get(orig_spk, orig_spk)) 
                        for (start, end, orig_spk) in merged_segments]
    # Merge the updated segments again based on the new speaker labels
    new_merged_segments = merge_segments(updated_segments, gap_threshold=1)
    
    # Prepare content for writing to the log file
    txt_lines = []
    txt_lines.append("Speaker Energies:")
    for spk in sorted(avg_energy.keys()):
        mapped_label = speaker_mapping.get(spk, spk)
        txt_lines.append(f"Original: {spk}, Mapped: {mapped_label}, Energy: {avg_energy[spk]:.4f}")
    txt_lines.append("\nSegment Information:")
    
    # copy to mute
    audio_mute = np.copy(audio)
    

    speaker0_counter = 0
    # Iterate over the newly merged segments, write the result info to the log,
    # and process the audio (mute non-target parts and save speaker 0 segments)
    for seg in new_merged_segments:
        start, end, mapped_label = seg
        # Discard the segment if its duration is less than 1 second
        if (end - start) < 1.0:
            continue

        start_str = sec_to_hhmmss(start)
        end_str = sec_to_hhmmss(end)
        line = f"{start_str} - {end_str} {mapped_label};"
        
        # If the speaker is speaker 0, save the segment as an individual .wav file
        # and record the filename in the log text
        if mapped_label == "speaker 0":
            filename = os.path.join(seg_save_path, "{:06d}.wav".format(speaker0_counter))
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment_audio = audio[start_sample:end_sample]
            sf.write(filename, segment_audio, sr)
            line += " [Saved as {:06d}.wav]".format(speaker0_counter)
            speaker0_counter += 1
        else:
            # Mute segments that do not belong to speaker 0
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            audio_mute[start_sample:end_sample] = 0.0
            
        txt_lines.append(line)

    # write log
    with open(log_save_path, "w", encoding="utf-8") as f:
        for line in txt_lines:
            f.write(line + "\n")
    
    # Save the muted audio to mute.wav
    sf.write(mute_save_path, audio_mute, sr)
