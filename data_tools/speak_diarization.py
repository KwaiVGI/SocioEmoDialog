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
    将秒数转换为 hh:mm:ss.ms 格式
    """
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def merge_segments(segments, gap_threshold=0.1):
    """
    合并相邻且连续（间隔小于 gap_threshold 秒）的同一说话人分段
    """
    segments = sorted(segments, key=lambda x: x[0])
    merged = []
    for seg in segments:
        start, end, speaker = seg
        if merged and speaker == merged[-1][2] and start - merged[-1][1] < gap_threshold:
            # 与上一个段同一说话人且间隔很短，则合并
            merged[-1] = (merged[-1][0], end, speaker)
        else:
            merged.append(seg)
    return merged

def compute_speaker_energy(audio, sr, segments):
    """
    根据每个分段的 RMS 能量（加权时间长度）计算每个说话人的平均能量
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
        # 计算 RMS 能量
        rms = np.sqrt(np.mean(segment_audio**2))
        duration = end - start
        speaker_energy[speaker] = speaker_energy.get(speaker, 0) + rms * duration
        speaker_duration[speaker] = speaker_duration.get(speaker, 0) + duration
    # 计算平均能量（加权）
    avg_energy = {spk: speaker_energy[spk] / speaker_duration[spk] for spk in speaker_energy}
    return avg_energy

def merge_similar_speakers(avg_energy, threshold=0.1):
    """
    将能量差异较小的speaker聚类。  
    参数：
      avg_energy：字典，key为原speaker标签，value为平均能量。
      threshold：如果两个speaker的平均能量之差小于该阈值（绝对值），则认为它们相似。
    返回：
      clusters：列表，每个元素是一个聚类，聚类内包含 (speaker, energy) 元组。
    """
    # 按平均能量从小到大排序
    sorted_speakers = sorted(avg_energy.items(), key=lambda x: x[1])
    clusters = []
    for spk, energy in sorted_speakers:
        if not clusters:
            clusters.append([(spk, energy)])
        else:
            # 与当前聚类最后一个speaker比较，采用绝对差异判断
            _, last_energy = clusters[-1][-1]
            if abs(energy - last_energy) < threshold:
                clusters[-1].append((spk, energy))
            else:
                clusters.append([(spk, energy)])
    return clusters

def assign_merged_labels(avg_energy, merge_threshold=0.1):
    """
    根据能量聚类结果对原始speaker标签进行重新映射，
    合并能量相近的speaker，并根据平均能量从高到低排序后分配新标签：
      能量最高的为 speaker 0，次高为 speaker 1，依次类推。
    """
    clusters = merge_similar_speakers(avg_energy, threshold=merge_threshold)
    # 对每个聚类计算平均能量
    cluster_info = []
    for cluster in clusters:
        speakers = [spk for spk, _ in cluster]
        # 聚类平均能量
        cluster_avg = np.mean([energy for _, energy in cluster])
        cluster_info.append((speakers, cluster_avg))
    
    # 按聚类平均能量从高到低排序
    cluster_info = sorted(cluster_info, key=lambda x: x[1], reverse=True)
    
    speaker_mapping = {}
    for new_label, (speakers, _) in enumerate(cluster_info):
        for spk in speakers:
            # 新的标签格式为 "speaker X"
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
    
    # 初步合并相邻连续段（gap_threshold 根据实际情况设置，例如这里用1秒）
    merged_segments = merge_segments(segments, gap_threshold=1)
    
    # 加载整个音频文件以便计算能量
    audio, sr = librosa.load(audio_file, sr=None)
    
    # 计算每个原始speaker的平均 RMS 能量
    avg_energy = compute_speaker_energy(audio, sr, merged_segments)
    
    # 根据能量差异（绝对值小于0.1）合并相近的speaker
    speaker_mapping = assign_merged_labels(avg_energy, merge_threshold=0.02)
    
    # 根据新的speaker_mapping更新每个分段的标签
    updated_segments = [(start, end, speaker_mapping.get(orig_spk, orig_spk)) 
                        for (start, end, orig_spk) in merged_segments]
    # 再次对更新后的分段按新标签进行合并
    new_merged_segments = merge_segments(updated_segments, gap_threshold=1)
    
    # 准备写入文本文件的内容
    txt_lines = []
    txt_lines.append("Speaker Energies:")
    for spk in sorted(avg_energy.keys()):
        mapped_label = speaker_mapping.get(spk, spk)
        txt_lines.append(f"Original: {spk}, Mapped: {mapped_label}, Energy: {avg_energy[spk]:.4f}")
    txt_lines.append("\nSegment Information:")
    
    # 复制原音频用于静音处理
    audio_mute = np.copy(audio)
    
    # 用于保存 speaker0 分段的编号计数
    speaker0_counter = 0
    
    # 遍历新的合并分段，将结果信息保存到文本中，并处理音频（静音以及保存 speaker0 段）
    for seg in new_merged_segments:
        start, end, mapped_label = seg
        # 如果分段时长小于1秒，则直接丢弃
        if (end - start) < 1.0:
            continue

        start_str = sec_to_hhmmss(start)
        end_str = sec_to_hhmmss(end)
        line = f"{start_str} - {end_str} {mapped_label};"
        
        # 如果是 speaker 0，则保存为独立的 wav 文件，并在文本中记录文件名
        if mapped_label == "speaker 0":
            filename = os.path.join(seg_save_path, "{:06d}.wav".format(speaker0_counter))
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment_audio = audio[start_sample:end_sample]
            sf.write(filename, segment_audio, sr)
            line += " [Saved as {:06d}.wav]".format(speaker0_counter)
            speaker0_counter += 1
        else:
            # 非 speaker 0 部分静音处理
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            audio_mute[start_sample:end_sample] = 0.0
            
        txt_lines.append(line)

    # 将文本内容写入 log 文件
    with open(log_save_path, "w", encoding="utf-8") as f:
        for line in txt_lines:
            f.write(line + "\n")
    
    # 保存静音处理后的音频到 mute.wav
    sf.write(mute_save_path, audio_mute, sr)
