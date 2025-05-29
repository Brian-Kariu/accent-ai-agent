import asyncio
import os

import torch
import torchaudio
import yt_dlp as youtube_dl
from speechbrain.inference import EncoderClassifier

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10 MB

# Load the pretrained ECAPA-TDNN model for accent identification (load once)
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    accent_identifier = EncoderClassifier.from_hparams(
        source="Jzuluaga/accent-id-commonaccent_ecapa",
        savedir="pretrained_models/accent-id-commonaccent_ecapa",
    )
except Exception as e:
    print(f"Error loading the model: {e}")
    accent_identifier = None


async def download_audio(url, output_path):
    """Downloads audio from a public URL asynchronously."""
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
    }
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, lambda: youtube_dl.YoutubeDL(ydl_opts).download([url])
        )
        return output_path
    except Exception as e:
        return f"Error downloading audio from URL: {e}"


async def convert_to_audio(video_path, audio_output_path="audio.wav"):
    """Converts a video file to an audio file using ffmpeg-python asynchronously."""
    try:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-i",
            video_path,
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            audio_output_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            return f"FFmpeg error: {stderr.decode('utf8')}"
        return audio_output_path
    except Exception as e:
        return f"Error during FFmpeg conversion: {e}"


async def process_url_for_audio(url, output_dir):
    """Downloads and converts audio, checks size, and uploads to Google Drive."""
    video_output_path_base = os.path.join(output_dir, "temp_video")
    audio_output_path = os.path.join(output_dir, "audio.wav")

    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": video_output_path_base,
        }
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, lambda: youtube_dl.YoutubeDL(ydl_opts).download([url])
        )

        import glob

        video_files = glob.glob(f"{video_output_path_base}*")
        if not video_files:
            return "Error: Could not find downloaded video file."
        video_file_path = video_files[0]

        audio_file_path = await convert_to_audio(video_file_path, audio_output_path)
        if "Error" in audio_file_path:
            os.remove(video_file_path)
            return audio_file_path

        # Check audio file size
        audio_file_size = os.path.getsize(audio_file_path)
        if audio_file_size > MAX_AUDIO_SIZE:
            os.remove(video_file_path)
            os.remove(audio_file_path)
            return f"Error: Processed audio file size exceeds the limit of {MAX_AUDIO_SIZE / (1024 * 1024):.1f} MB."

        os.remove(video_file_path)
        return audio_file_path
    except Exception as e:
        return f"Error processing URL for audio: {e}"


async def predict_accent(audio_path):
    """Predicts the accent from an audio file."""
    if accent_identifier is None:
        return {"error": "Accent identification model not loaded."}

    try:
        signal, sampling_rate = torchaudio.load(audio_path)
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sampling_rate, new_freq=16000
            )
            signal = resampler(signal)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)  # Convert to mono

        output = accent_identifier.classify_batch(signal)
        logits = output[0][0]
        predicted_index = torch.argmax(logits).item()
        labels = accent_identifier.hparams.label_encoder.ind2lab
        predicted_accent = labels[predicted_index]
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        confidence = probabilities[predicted_index].item() * 100

        results = {"accent": predicted_accent, "confidence": f"{confidence:.2f}%"}
        return results
    except Exception as e:
        return {"error": f"Error during accent prediction: {e}"}
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
