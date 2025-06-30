from preprocess import create_embeddings

from pathlib import Path

synthesizer_root = Path(r"C:\Users\soham\FInalVoice Cloning\LibriSpeech\processed\synthesizer")
encoder_model_fpath = Path(r"C:\Users\soham\OneDrive\Documents\COntinje VOICECLONING\Real-Time-Voice-Cloning\saved_models\default\encoder.pt")
create_embeddings(
    synthesizer_root=synthesizer_root,
    encoder_model_fpath=encoder_model_fpath,
)
