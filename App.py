import streamlit as st
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Union
from loader import load_voice_cloning_models
# Streamlit config
st.set_page_config(layout="centered")



st.markdown("""
        <style>
        body {
            background-image: url('https://www.google.com/url?sa=i&url=https%3A%2F%2Fstartupsmagazine.co.uk%2Farticle-voice-cloning-startup-raises-80-million&psig=AOvVaw2oLqzvU0l0ybV5E_d7W7Wg&ust=1751379504187000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCPC56qyrmY4DFQAAAAAdAAAAABAE');
            background-size: cover;
            background-position: center;
            color: white;
        }
        .stTextInput > div > div > input {
            background-color: #f0fff0;
            color: black;
        }
        .stFileUploader, .stTextInput, .stDownloadButton {
            border-radius: 12px;
        }
        .css-1d391kg {
            background-color: rgba(0,0,0,0.5);
            padding: 20px;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align:center; color:#90EE90;'>CLONE VOICE WITH YOUR TEXT</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:right;'>— Developed by Soham Kale</p>", unsafe_allow_html=True)

# Import voice cloning modules
from Encoder.audio import preprocess_wav, wav_to_mel_spectrogram
from Encoder.model import SpeakerEncoder
from Tacotron.model import Tacotron
from Tacotron.utils.text import text_to_sequence, symbols
from wavernn.model import WaveRNN

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def get_model_paths():
    return load_voice_cloning_models()
def load_models():
    model_paths = get_model_paths()    
    enc = SpeakerEncoder(device, device)
    enc.load_state_dict(torch.load(model_paths["encoder.pt"], map_location=device)["model_state"])

    syn = Tacotron(
        embed_dims=512,
        num_chars=len(symbols),
        encoder_dims=256,
        decoder_dims=128,
        n_mels=80,
        fft_bins=80,
        postnet_dims=512,
        encoder_K=5,
        lstm_dims=1024,
        postnet_K=5,
        num_highways=4,
        dropout=0.5,
        stop_threshold=-3.4,
        speaker_embedding_size=256
    ).to(device)
    syn.load_state_dict(torch.load(model_paths["synthesizer.pt"], map_location=device)["model_state"])

    voc = WaveRNN(
        rnn_dims=512,
        fc_dims=512,
        bits=9,
        pad=2,
        upsample_factors=(5, 5, 8),
        feat_dims=80,
        compute_dims=128,
        res_out_dims=128,
        res_blocks=10,
        hop_length=200,
        sample_rate=16000,
    )
    voc.load_state_dict(torch.load(model_paths["vocoder.pt"], map_location=device)["model_state"])

    return enc, syn, voc

# Load models
model1, model2, model3 = load_models()

# Helper functions
def pad1d(x, max_len, pad_value=0):
    return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)

def synthesize_spectrograms(texts: List[str],
                                embeddings: Union[np.ndarray, List[np.ndarray]],
                                return_alignments=False):
       
        inputs = [text_to_sequence(text.strip(), ["english_cleaners"]) for text in texts]
        if not isinstance(embeddings, list):
            embeddings = [embeddings]

        batched_inputs = [inputs[i:i+16]
                             for i in range(0, len(inputs), 16)]
        batched_embeds = [embeddings[i:i+16]
                             for i in range(0, len(embeddings), 16)]

        specs = []
        for i, batch in enumerate(batched_inputs, 1):
            if True:
                print(f"\n| Generating {i}/{len(batched_inputs)}")

            text_lens = [len(text) for text in batch]
            max_text_len = max(text_lens)
            chars = [pad1d(text, max_text_len) for text in batch]
            chars = np.stack(chars)

            speaker_embeds = np.stack(batched_embeds[i-1])

            chars = torch.tensor(chars).long().to(device)
            speaker_embeddings = torch.tensor(speaker_embeds).float().to(device)

            # Inference
            _, mels, alignments = model2.generate(chars, speaker_embeddings)
            mels = mels.detach().cpu().numpy()
            for m in mels:
                # Trim silence from end of each spectrogram
                while np.max(m[:, -1]) < -3.4:
                    m = m[:, :-1]
                specs.append(m)

        if True:
            print("\n\nDone.\n")
        return (specs, alignments) if return_alignments else specs
def embed_frames_batch(frames_batch):
  
    

    frames = torch.from_numpy(frames_batch).to(device)
    embed = model1.forward(frames).detach().cpu().numpy()
    return embed


def compute_partial_slices(n_samples, partial_utterance_n_frames=160,
                           min_pad_coverage=0.75, overlap=0.5):
    
    assert 0 <= overlap < 1
    assert 0 < min_pad_coverage <= 1

    samples_per_frame = int((16000 * 10 / 1000))
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

    # Compute the slices
    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + partial_utterance_n_frames])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))

    # Evaluate whether extra padding is warranted or not
    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
    if coverage < min_pad_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]

    return wav_slices, mel_slices

def embed_utterance(wav, using_partials=True, return_partials=False, **kwargs):

    if not using_partials:
        frames = wav_to_mel_spectrogram(wav)
        embed = embed_frames_batch(frames[None, ...])[0]
        if return_partials:
            return embed, None, None
        return embed

    # Compute where to split the utterance into partials and pad if necessary
    wave_slices, mel_slices = compute_partial_slices(len(wav), **kwargs)
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

    # Split the utterance into partials
    frames = wav_to_mel_spectrogram(wav)
    frames_batch = np.array([frames[s] for s in mel_slices])
    partial_embeds = embed_frames_batch(frames_batch)

    # Compute the utterance embedding from the partial embeddings
    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)

    if return_partials:
        return embed, partial_embeds, wave_slices
    return embed

# User Input
uploaded_file = st.file_uploader("Upload your reference voice (.wav)", type=["wav"])
text_input = st.text_input("Enter your text here", placeholder="Type something to clone the voice")

if uploaded_file and text_input:
    st.audio(uploaded_file, format="audio/wav")
    
    with st.spinner("⏳ Cloning voice... Please wait."):
        # Save uploaded file
        input_path = Path("audio/test.wav")
        input_path.parent.mkdir(exist_ok=True)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        # Process and clone
        preprocessed_wav = preprocess_wav(input_path)
        original_embed = embed_utterance(preprocessed_wav)
        mel = synthesize_spectrograms([text_input], [original_embed])[0]
        mel_tensor = torch.from_numpy((mel / 4.)[None, ...])
        wav = model3.generate(mel_tensor, batched=True, target=8000, overlap=800, mu_law=True)

        # Save output
        output_path = Path("audio/output.wav")
        sf.write(output_path, wav.astype(np.float32), 16000)

        # Embed cloned output
        cloned_embed = embed_utterance(wav)
        similarity = cosine_similarity([original_embed], [cloned_embed])[0][0] * 100

    # Show results
    st.success("✅ Voice cloned!")
    st.audio(str(output_path), format="audio/wav")

    # 🎧 Download button
    with open(output_path, "rb") as f:
        st.download_button(
            label="🎧 Download Cloned Voice",
            data=f,
            file_name="cloned_voice.wav",
            mime="audio/wav"
        )

    # 📈 Show spectrogram
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mel, aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)
    ax.set_title("Mel Spectrogram")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel Channels")
    st.pyplot(fig)

    # 🎙️ Speaker similarity
    st.markdown(f"🧬 **Speaker Similarity Score:** `{similarity:.2f}%`")

 
    
