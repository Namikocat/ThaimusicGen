  # SplitSong.py
import torchaudio
import os

input_path = ""
output_dir = ""
os.makedirs(output_dir, exist_ok=True)

waveform, sr = torchaudio.load(input_path)
waveform = torchaudio.transforms.Resample(sr, 32000)(waveform)
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

segment_duration = 15
samples_per_segment = 32000 * segment_duration
total_samples = waveform.shape[1]

basename = "song_clipO"
for i in range(0, total_samples, samples_per_segment):
    clip = waveform[:, i:i + samples_per_segment]
    if clip.shape[1] < samples_per_segment:
        continue
    out_path = os.path.join(output_dir, f"{basename}_{i // samples_per_segment}.wav")
    torchaudio.save(out_path, clip, 32000)
