import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout


class DescriptiveAudioDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, segment_duration=30, sample_rate=22050):
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.audio_dir = audio_dir

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        audio_path = os.path.join(self.audio_dir, item['audio'])
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
        target_len = self.sample_rate * self.segment_duration
        if waveform.shape[1] < target_len:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))
        else:
            waveform = waveform[:, :target_len]
        return waveform, item['description']


def custom_collate(batch):
    waveforms, descriptions = zip(*batch)
    return torch.stack(waveforms), list(descriptions)


class MusicGenFinetuning(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MusicGen.get_pretrained("facebook/musicgen-medium")
        self.model.lm.train()
        self.model.lm = self.model.lm.float()

    def training_step(self, batch):
        waveforms, descriptions = batch

        with torch.no_grad():
            codes, _ = self.model.compression_model.encode(waveforms)

        attributes, _ = self.model._prepare_tokens_and_attributes(descriptions, None)
        condition_tensors = get_condition_tensor(self.model, attributes)

        lm_output = self.model.lm.compute_predictions(
            codes=codes, conditions=[], condition_tensors=condition_tensors
        )

        logits, mask = lm_output.logits[0], lm_output.mask[0].view(-1)
        codes = F.one_hot(codes[0], 2048).float()
        masked_logits = logits.view(-1, 2048)[mask]
        masked_codes = codes.view(-1, 2048)[mask]
        loss = F.cross_entropy(masked_logits, masked_codes)

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.lm.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=0.1)


def get_condition_tensor(model, attributes):
    tokenized = model.lm.condition_provider.tokenize(attributes)
    return model.lm.condition_provider(tokenized)


if __name__ == "__main__":
    L.seed_everything(42)

    metadata_file = "/teamspace/studios/this_studio/segments3-new/segments3/data.json"
    audio_dir = "/teamspace/studios/this_studio/segments3-new/segments3"

    dataset_train = DescriptiveAudioDataset(metadata_file, audio_dir)
    train_dataloader = DataLoader(dataset_train, batch_size=1, collate_fn=custom_collate, num_workers=4)

    model = MusicGenFinetuning()

    # ตั้งค่า Checkpoint สำหรับบันทึกโมเดลอัตโนมัติ
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="musicgen-{epoch:02d}-{train_loss:.2f}",
        save_top_k=1,
        monitor="train_loss",
        mode="min"
    )

    trainer = L.Trainer(
        precision="16-mixed",
        max_epochs=10,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_dataloader)

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.model.lm.state_dict(), "saved_models/finetuned_musicgen_lm3.pt")
    print("✅ โมเดลถูกบันทึกไว้ที่: saved_models/finetuned_musicgen_lm3.pt")
