
import audio
from pathlib import Path
from tqdm import tqdm
import numpy as np
import librosa
import os
import torch
import sys
sys.path.append(os.path.abspath(r"C:\Users\soham\FInalVoice Cloning"))
from Encoder import audio as encoderAudio
from Encoder.model import SpeakerEncoder

def preprocess_dataset(datasets_root: Path, out_dir: Path,hparams,
                        datasets_name: str):
    
    if not os.path.exists(datasets_root):
        raise FileNotFoundError(f"Encoder model file {datasets_root} does not exist.")

    out_dir.joinpath("mels").mkdir(exist_ok=True)
    out_dir.joinpath("audio").mkdir(exist_ok=True)

    
    metadata_fpath = out_dir.joinpath("train.txt")
    metadata_file = metadata_fpath.open("w", encoding="utf-8")

    speaker_dirs = list(datasets_root.glob("*"))
    for speaker_dir in tqdm(speaker_dirs, desc=f"Processing {datasets_name}", unit="speaker"):
        if not speaker_dir.is_dir():
            continue
        speaker_metadata = preprocess_speaker(
            speaker_dir=speaker_dir,
            out_dir=out_dir,
            hparams=hparams
        )
        for metadatum in speaker_metadata:
            metadata_file.write("|".join(str(x) for x in metadatum) + "\n")

    metadata_file.close()

    with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
        metadata = [line.strip().split("|") for line in metadata_file]

    mel_frames = sum(int(m[4]) for m in metadata)
    timesteps = sum(int(m[3]) for m in metadata)
    hours = (timesteps / hparams.sample_rate) / 3600

    print(f"\n✅ Dataset: {len(metadata)} utterances")
    print(f"🔹 {mel_frames} mel frames")
    print(f"🔹 {timesteps} audio timesteps ({hours:.2f} hours)")
    print(f"🔹 Max input text length: {max(len(m[5]) for m in metadata)}")
    print(f"🔹 Max mel length: {max(int(m[4]) for m in metadata)}")
    print(f"🔹 Max audio timesteps: {max(int(m[3]) for m in metadata)}")



def preprocess_speaker(speaker_dir: Path,out_dir: Path,hparams):
    metadata = []

    for book_dir in speaker_dir.glob("*"):
        if not book_dir.is_dir():
            continue

     
        text_file_path = next(book_dir.glob("*.txt"), None)
        if text_file_path is None:
            print(f"❌ No text file found in {book_dir}, skipping.")
            continue

     
        text_dict = {}
        with text_file_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) != 2:
                    continue
                basename, text = parts
                text_dict[basename] = text.replace("\"", "").strip()

        for wav_fpath in book_dir.glob("*.flac"):
            basename = wav_fpath.stem
            if basename not in text_dict:
                print(f"⚠️ No transcription found for {basename}")
                continue

            wav, _ = librosa.load(str(wav_fpath), sr=hparams.sample_rate)
            if hparams.rescale:
                wav = wav / np.abs(wav).max() * hparams.rescaling_max

            
            text = text_dict[basename]
            item = process_utterance(wav, text, out_dir, basename,hparams)
            if item is not None:
                metadata.append(item)

    return metadata

def process_utterance(wav: np.ndarray, text: str, out_dir: Path, basename: str,
                    hparams):
    
    mel_fpath = out_dir.joinpath("mels", "mel-%s.npy" % basename)
    wav_fpath = out_dir.joinpath("audio", "audio-%s.npy" % basename)

    if hparams.trim_silence:
        wav = encoderAudio.preprocess_wav(wav, normalize=False, trim_silence=True)

    if len(wav) < hparams.utterance_min_duration * hparams.sample_rate:
        return None

    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        return None

    np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
    np.save(wav_fpath, wav, allow_pickle=False)

    return wav_fpath.name, mel_fpath.name, "embed-%s.npy" % basename, len(wav), mel_frames, text



def create_embeddings(synthesizer_root: Path, encoder_model_fpath: Path):
  
    wav_dir = synthesizer_root / "audio"
    metadata_fpath = synthesizer_root / "train.txt"
    embed_dir = synthesizer_root / "embeds"
    embed_dir.mkdir(exist_ok=True)

    assert wav_dir.exists(), "Audio directory not found."
    assert metadata_fpath.exists(), "train.txt metadata file not found."

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model =SpeakerEncoder(device, torch.device("cpu"))
    checkpoints=torch.load(encoder_model_fpath,map_location=device)
    model.load_state_dict(checkpoints["model_state"])
    model.eval()
   
    with metadata_fpath.open("r") as f:
        metadata = [line.strip().split("|") for line in f]

    for wav_name, _, embed_name, *_ in tqdm(metadata, desc="Embedding", unit="utterance"):
        wav_fpath = wav_dir / wav_name
        embed_fpath = embed_dir / embed_name

        
        wav = np.load(wav_fpath)

        preprocessed_wav = encoderAudio.preprocess_wav(wav)

        
        frames = encoderAudio.wav_to_mel_spectrogram(preprocessed_wav)
        
        frames = torch.from_numpy(frames[None, ...]).to(device)
        embed =  model.forward(frames).detach().cpu().numpy()[0]

        np.save(embed_fpath, embed, allow_pickle=False)


