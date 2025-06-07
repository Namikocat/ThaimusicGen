import torch
from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-medium") #โหลดโมเดลmusicgen


model.lm.load_state_dict(torch.load("saved_models/finetuned_musicgen_lm3.pt")) #โหลดโมเดลที่ finetune ไว้
model.lm.eval() 


descriptions = ["Thai song with saw u"] #prompt

model.set_generation_params(duration=10)  # ความยาวเสียง (วินาที)
waveforms = model.generate(descriptions)

torchaudio.save("generated_music4.wav", waveforms[0].cpu(), sample_rate=32000)
print("✅ เสียงถูกสร้างและบันทึกไว้ที่: generated_music4.wav")
