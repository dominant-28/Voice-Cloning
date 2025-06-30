from pathlib import Path
from hparams import hparams
from train import train



if __name__ == "__main__":

    run_id = "tacotron_training"
    syn_dir = Path(r"C:\Users\soham\FInalVoice Cloning\LibriSpeech\processed\synthesizer")  
    models_dir = Path(r"C:\Users\soham\FInalVoice Cloning\Tacotron\utils\saved_model")
    save_every = 1000
    backup_every = 25000
    force_restart = False
    hparams_str = "" 

    parsed_hparams = hparams.parse(hparams_str)

    args = {
        "run_id": run_id,
        "syn_dir": syn_dir,
        "models_dir": models_dir,
        "force_restart": force_restart,
        "hparams": parsed_hparams
    }

    train(**args)
