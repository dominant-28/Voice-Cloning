#  Voice Cloning using Deep Learning
###  Demo  
[![Streamlit App](https://img.shields.io/badge/Streamlit-Demo-red?logo=streamlit)](https://voice-cloning-sk-28.streamlit.app/)
#### This project implements a real-time **Voice Cloning System** that can synthesize human-like speech from text using a short voice sample of any speaker. The system is built using a deep learning pipeline composed of three core models:

###  Pipeline Overview

1. **Speaker Encoder**  
   Extracts a fixed-dimensional speaker embedding from a short audio clip of the target speaker. This embedding captures the unique vocal characteristics of the speaker.

2. **Tacotron**  
   A sequence-to-sequence model that generates a mel-spectrogram from the given text and speaker embedding. It learns the pronunciation and prosody patterns of the target voice.

3. **WaveRNN**  
   A neural vocoder that converts the generated mel-spectrogram into an audio waveform, producing realistic and high-fidelity speech output.

---

###  Dataset

- The system is trained and tested using the **LibriSpeech** dataset.
- It supports cloning from both predefined dataset samples and user-uploaded voice clips.

## Tacotron: Text-to-Spectrogram Model

Tacotron is a sequence-to-sequence model that converts input text into mel-spectrograms, which are then used by a neural vocoder (e.g., WaveRNN) to generate raw audio. It is composed of an encoder-decoder architecture with an attention mechanism.

One of the key components of Tacotron is the **CBHG module**, which is used both in the encoder and the post-processing stage of the decoder.

---

###  Tacotron Architecture

![Tacotron Architecture](https://drive.google.com/uc?export=view&id=1BWcAj0ooLchqHnH0oUARYPh2a-t_hxNM)

1. **Encoder**  
   - **Character Embeddings**: Input text is converted into embedded vectors.
   - **Pre-net**: Two fully connected layers with ReLU activations and dropout. Helps with generalization.
   - **CBHG Module**: A powerful feature extractor that transforms the embedded input into high-level representations using convolutional, highway, and bidirectional recurrent layers.

2. **Attention**  
   - Aligns encoder outputs with decoder inputs.  
   - Allows the model to focus on the relevant parts of the input text during each step of decoding.

3. **Decoder**  
   - **Pre-net + RNN**: Takes the previous mel-spectrogram frame (or a <GO> token initially) as input and predicts the next frame(s).
   - **Post-processing CBHG**: Refines the raw decoder outputs into a cleaner, more accurate spectrogram.

---

##  CBHG Module (Used in Encoder and Post-Net)

![CBHG Module](https://drive.google.com/uc?export=view&id=1Mn7Qxx0qGJxsOEhNgWmALptz5JaaG8eP)

The **CBHG module** is a core block in Tacotron used for extracting robust sequential features from text and refining spectrogram predictions.

### 🔧 Structure:

- **Convolution Bank**: A series of 1D convolutions with varying kernel sizes to capture local patterns at different scales.
- **Max Pooling**: Preserves sequence length while emphasizing the most salient features.
- **Conv1D Projections**: Applies linear and ReLU convolutions to project features.
- **Highway Networks**: Fully connected layers with gating mechanisms to control information flow.
- **Bidirectional GRU (Bi-GRU)**: Captures long-range dependencies in both forward and backward directions.

###  Usage in Tacotron:

- **Encoder CBHG**: Converts character-level input to sequential embeddings with context.
- **Decoder Post-CBHG**: Refines coarse spectrogram outputs from the decoder into sharper and more accurate representations.

---

#### By combining the strengths of attention mechanisms, RNNs, and the CBHG module, Tacotron is able to generate natural-sounding speech with expressive prosody and accurate pronunciation from text input.
---

##  WaveRNN: Neural Vocoder for Audio Waveform Generation

**WaveRNN** is a powerful and efficient neural vocoder used to convert mel-spectrograms into high-fidelity raw audio waveforms. Unlike traditional vocoders, WaveRNN generates audio sample-by-sample, producing natural and realistic speech even at high quality.

---

###  Architecture Overview

WaveRNN consists of the following main components:

1. **1D ResNet (Residual Blocks)**  
   - Extracts high-level temporal features from input mel-spectrograms and previous audio samples.
   - A series of residual convolutional blocks (ResBlock 1 to N) with skip connections help retain essential features across layers.

2. **Upsampling Network**  
   - Mel-spectrograms are temporally upsampled using 2D convolution layers followed by nearest-neighbor upsampling.
   - Aligns the time resolution of mel-spectrograms with that of the waveform.

3. **Autoregressive Core**  
   - Composed of GRU layers followed by dense layers.
   - Predicts one audio sample at a time based on previous samples and mel features.
   - Uses 9-bit quantized output or similar schemes for efficient computation.

4. **Output Layers**  
   - Final fully connected layers that generate audio sample values one-by-one.

---

### Model Flow

- Inputs: A sequence of mel-spectrogram frames and previously generated waveform samples.
- Processing: The features are passed through a ResNet and upsampling stack, then split and processed by GRUs.
- Outputs: Raw waveform samples generated autoregressively.

---

### 📈 Diagram: WaveRNN Model Overview

![WaveRNN Architecture](https://drive.google.com/uc?export=view&id=1IWoLf-ro0nGyfnyPDTQpjK3TEq5HbjeB)

> The diagram illustrates the data flow through the WaveRNN pipeline, from mel-spectrogram input to audio waveform output.

---

###  Key Features

- High-quality audio synthesis with low computational cost.
- Can run in real-time on consumer-grade hardware.
- Ideal for deployment in TTS and voice cloning applications.

WaveRNN is the final stage in the voice cloning pipeline, taking the mel-spectrogram output from Tacotron and generating realistic, human-like speech.

---


## 🧪 Integration

This project integrates three key components into a complete voice cloning pipeline:

- **Speaker Encoder**: Extracts speaker-specific embeddings from short voice samples.
- **Tacotron**: Converts input text and speaker embeddings into mel-spectrograms with natural prosody.
- **WaveRNN**: Transforms the mel-spectrograms into high-quality, human-like audio waveforms.

Together, these models enable end-to-end voice cloning from just a few seconds of audio.

## Getting Started
 
### Prerequisites
 
- Python 3.9 or 3.10
- pip
- (Recommended) A CUDA-capable GPU — the notebook falls back to CPU automatically if none is found
- ffmpeg (recommended for reading/writing various audio formats)
### 1. Clone the repository
 
```bash
git clone https://github.com/dominant-28/Voice-Cloning.git
cd Voice-Cloning
```
 
### 2. Create a Python virtual environment
 
**Using venv:**
 
```bash
python -m venv venv
 
# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```
 
**Or using conda:**
 
```bash
conda create -n voice-cloning python=3.10 -y
conda activate voice-cloning
```
 
### 3. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
This installs: `numpy`, `librosa`, `torch`, `torchvision`, `torchaudio`, `tqdm`, `scipy`, `matplotlib`, `scikit-learn`, `soundfile`, `webrtcvad`, `streamlit`, `unidecode`, and `gdown`.
 
> Note: the `requirements.txt` doesn't pin a specific `torch` build. If you have a GPU, install the CUDA-matched version of `torch`/`torchaudio` from [pytorch.org](https://pytorch.org/get-started/locally/) *before* running `pip install -r requirements.txt`, so pip doesn't overwrite it with a CPU-only build.
 
Also install Jupyter if it's not already available:
 
```bash
pip install jupyter notebook
```
 
### 4. Download the pretrained checkpoints
 
The notebook loads three pretrained checkpoints:
 
- `encoder.pt` – Speaker Encoder weights
- `synthesizer.pt` – Tacotron synthesizer weights
- `vocoder.pt` – WaveRNN vocoder weights
Place these files somewhere on disk (e.g. a `checkpoints/` folder in the project root), and update the paths in `generator.ipynb` accordingly (see next section).
 
> If you have a shared/hosted copy of these checkpoints (e.g. on Google Drive), you can use `gdown` (already in `requirements.txt`) to fetch them, or add direct download instructions/links here once you have a hosting location for them.
 
## Usage
 
1. Launch the notebook:
```bash
   jupyter notebook generator.ipynb
```
 
2. **Update the checkpoint paths** near the top of the notebook to point to your local files:
```python
   checkpoint1 = torch.load(r"path/to/encoder.pt", map_location=device)
   checkpoint2 = torch.load(r"path/to/synthesizer.pt", map_location=device)
   checkpoint3 = torch.load(r"path/to/vocoder.pt", map_location=device)
```
 
3. **Set your reference audio and text**, then run the generation cells:
```python
   input_file = Path(r"path/to/reference_audio.wav")
   preprocessed_wav = preprocess_wav(input_file)
   embed = embed_utterance(preprocessed_wav)
 
   text = "Type the sentence you want spoken in the cloned voice here."
 
   mel = synthesize_spectrograms([text], [embed])[0]
   mel = mel / 4.
   mel = torch.from_numpy(mel[None, ...])
   wav = model3.generate(mel, True, 8000, 800, True, None)
```
 
4. **Save the output:**
```python
   path = Path(r"path/to/output.wav")
   sf.write(path, wav.astype(np.float32), 16000)
```
 
The result is a `.wav` file of the input text spoken in the cloned voice, sampled at 16 kHz.
 
## Project Structure
 
```
Voice-Cloning/
├── Encoder/              # Speaker encoder model + audio preprocessing (preprocess_wav, wav_to_mel_spectrogram)
├── Tacotron/              # Synthesizer model + text utilities (symbols, text_to_sequence)
├── wavernn/               # WaveRNN vocoder model
├── generator.ipynb        # Main demo notebook — set audio path + text, run to generate cloned speech
├── requirements.txt       # Python dependencies
└── README.md
```
 
> Update this structure if your actual folder names/layout differ.
 
## Troubleshooting
 
- **`FileNotFoundError` on checkpoint load**: Double-check the `encoder.pt` / `synthesizer.pt` / `vocoder.pt` paths in the notebook — they currently use placeholder Windows-style paths (`r"DIR\..."`) and need to point to files that actually exist on your machine.
- **Slow generation on CPU**: This is expected — the WaveRNN vocoder in particular benefits a lot from a GPU. Install a CUDA-enabled `torch` build if you have a compatible GPU.
- **Poor voice similarity**: Use a clean reference clip (a few seconds, minimal background noise/music) for best speaker-embedding quality.
- **`webrtcvad` install issues**: On some systems this requires build tools (`build-essential` on Linux, or Visual C++ Build Tools on Windows) since it compiles a small C extension.

##  References

- [Tacotron: A Fully End-to-End Text-To-Speech Synthesis Model](https://arxiv.org/abs/1703.10135)
- [WaveRNN: A Compact, Efficient Neural Vocoder](https://arxiv.org/abs/1802.08435)
- [Real-Time Voice Cloning (Corentin Jemine)](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [LibriSpeech Dataset](https://www.openslr.org/12)

---

##  Contact

**Soham Kale**  
📧 [sohamkale2828@gmail.com]  




