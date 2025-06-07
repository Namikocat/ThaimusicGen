import torch
from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout
import torchaudio

# โหลด MusicGen พื้นฐาน
model = MusicGen.get_pretrained("facebook/musicgen-medium")

# โหลด LM ที่ผ่านการ finetune
model.lm.load_state_dict(torch.load("saved_models/finetuned_musicgen_lm3.pt"))
model.lm.eval()  # ใช้กับ language model ที่ฝึกไว้

# ใส่ข้อความบรรยายที่ต้องการสร้างเป็นเพลง
descriptions = ["Thai song with saw u"]

# สร้างเพลง (waveform)
model.set_generation_params(duration=10)  # ความยาวเสียง (วินาที)
waveforms = model.generate(descriptions)

# บันทึกเป็น .wav
torchaudio.save("generated_music4.wav", waveforms[0].cpu(), sample_rate=32000)
print("✅ เสียงถูกสร้างและบันทึกไว้ที่: generated_music4.wav")
