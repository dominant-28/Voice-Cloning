from preprocess import create_embeddings

from pathlib import Path

synthesizer_root = Path(r"DIR\LibriSpeech\processed\synthesizer")
encoder_model_fpath = Path(r"DIR\encoder.pt")
create_embeddings(
    synthesizer_root=synthesizer_root,
    encoder_model_fpath=encoder_model_fpath,
)
