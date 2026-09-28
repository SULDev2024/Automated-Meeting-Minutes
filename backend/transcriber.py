import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY was not found. Add it to the .env file."
    )

client = Groq(api_key=api_key)


def transcribe_audio(file_path: str):
    """
    Send an audio/video file to Groq for multilingual transcription.
    """
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(file_path), audio_file.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
            temperature=0.0,
            timestamp_granularities=["segment"],
        )

    segments = []
    if transcription.segments:
        for segment in transcription.segments:
            segments.append({
                "start": round(segment["start"], 2),
                "end": round(segment["end"], 2),
                "text": segment["text"].strip(),
            })

    return {
        "language": transcription.language,
        "segments": segments,
        "full_text": transcription.text.strip(),
    }
