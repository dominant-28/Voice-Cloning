from preprocess import preprocess_dataset
from hparams import hparams
from pathlib import Path

datasets_root = Path(r"C:\Users\soham\FInalVoice Cloning\LibriSpeech\train-clean-100")


out_dir = Path(r"C:\Users\soham\FInalVoice Cloning\LibriSpeech") / "processed" / "synthesizer"
datasets_name = "LibriSpeech"
out_dir.mkdir(exist_ok=True, parents=True)


preprocess_dataset(
    datasets_root=datasets_root,
    out_dir=out_dir,
    hparams=hparams,
    datasets_name=datasets_name,
   
)
