from pathlib import Path
import torch
from hparams import hparams
from tacotron_dataset import SynthesizerDataset,collate_synthesizer
from model import Tacotron
from utils.text import symbols
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from functools import partial
def generate_gta_mels(datasets_root=Path(r"C:\Users\soham\FInalVoice Cloning\LibriSpeech"), syn_model_fpath=Path(r"C:\Users\soham\OneDrive\Documents\COntinje VOICECLONING\Real-Time-Voice-Cloning\saved_models\default\synthesizer.pt")):
    
    in_dir = datasets_root / "processed" / "synthesizer"
    out_dir = datasets_root / "processed" / "vocoder"
    synth_dir = out_dir / "mels_gta"
    synth_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Tacotron(
        embed_dims=hparams.tts_embed_dims,
        num_chars=len(symbols),
        encoder_dims=hparams.tts_encoder_dims,
        decoder_dims=hparams.tts_decoder_dims,
        n_mels=hparams.num_mels,
        fft_bins=hparams.num_mels,
        postnet_dims=hparams.tts_postnet_dims,
        encoder_K=hparams.tts_encoder_K,
        lstm_dims=hparams.tts_lstm_dims,
        postnet_K=hparams.tts_postnet_K,
        num_highways=hparams.tts_num_highways,
        dropout=0.,
        stop_threshold=hparams.tts_stop_threshold,
        speaker_embedding_size=hparams.speaker_embedding_size,
    ).to(device)

    model.load(syn_model_fpath)
    model.eval()

    r = int(model.r)
    metadata_fpath = in_dir / "train.txt"
    mel_dir = in_dir / "mels"
    embed_dir = in_dir / "embeds"

    dataset = SynthesizerDataset(metadata_fpath, mel_dir, embed_dir, hparams)
    data_loader = DataLoader(dataset, batch_size=hparams.synthesis_batch_size,
                             collate_fn=partial(collate_synthesizer, r=r, hparams=hparams),
                             num_workers=0)

    with open(out_dir / "synthesized.txt", "w") as file:
        for texts, mels, embeds, idx in tqdm(data_loader):
            texts, mels, embeds = texts.to(device), mels.to(device), embeds.to(device)
            _, mels_out, _, _ = model(texts, mels, embeds)

            for j, k in enumerate(idx):
                mel_out = mels_out[j].detach().cpu().numpy().T
                mel_out = mel_out[:int(dataset.metadata[k][4])]
                np.save(synth_dir / dataset.metadata[k][1], mel_out)
                file.write("|".join(dataset.metadata[k]) + "\n")


if __name__ == "__main__":
    generate_gta_mels()
