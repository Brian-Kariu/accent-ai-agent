import os

import ffmpeg
import torch
import torchaudio
import yt_dlp as youtube_dl
from flask import Flask, jsonify, render_template, request
from speechbrain.pretrained import EncoderClassifier

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load the pretrained ECAPA-TDNN model for accent identification (load once)
try:
    accent_identifier = EncoderClassifier.from_hparams(
        source="Jzuluaga/accent-id-commonaccent_ecapa",
        savedir="pretrained_models/accent-id-commonaccent_ecapa",
    )
except Exception as e:
    print(f"Error loading the model: {e}")
    accent_identifier = None


def download_audio(url, output_path):
    """Downloads audio from a public URL."""
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    except Exception as e:
        return f"Error downloading audio from URL: {e}"


def convert_to_audio(video_path, audio_output_path="audio.wav"):
    """Converts a video file to an audio file using ffmpeg-python."""
    try:
        (
            ffmpeg.input(video_path)
            .output(audio_output_path, acodec="pcm_s16le", ac=1, ar=16000)
            .run(capture_stdout=True, capture_stderr=True)
        )
        return audio_output_path
    except ffmpeg.Error as e:
        return f"FFmpeg error: {e.stderr.decode('utf8')}"


def process_url_for_audio(url, output_dir):
    """Downloads and converts audio from a public URL."""
    video_output_path = os.path.join(output_dir, "temp_video.%(ext)s")
    audio_output_path = os.path.join(output_dir, "audio.wav")

    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": video_output_path,
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            if info_dict and "entries" in info_dict:
                # Handle playlists or multiple videos
                entry = info_dict["entries"][0]
                video_file_path = ydl.prepare_filename(entry)
            else:
                video_file_path = ydl.prepare_filename(info_dict)

            convert_to_audio(video_file_path, audio_output_path)
            os.remove(video_file_path)  # Clean up the video file
            return audio_output_path
    except Exception as e:
        return f"Error processing URL for audio: {e}"


def predict_accent(audio_path):
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


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/detect_accent", methods=["POST"])
def detect_accent_api():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' in request."}), 400

    url = data["url"]
    output_dir = app.config["UPLOAD_FOLDER"]
    audio_path_or_error = process_url_for_audio(url, output_dir)

    if isinstance(audio_path_or_error, str) and "Error" in audio_path_or_error:
        return jsonify({"error": audio_path_or_error}), 500

    audio_path = audio_path_or_error
    prediction_result = predict_accent(audio_path)
    return jsonify(prediction_result)


if __name__ == "__main__":
    app.run(debug=True)
