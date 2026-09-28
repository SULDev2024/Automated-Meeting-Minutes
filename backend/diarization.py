import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY was not found. Add it to the .env file."
    )

client = OpenAI(api_key=api_key)


def diarize_audio(file_path: str):
    """
    Transcribe audio and identify who spoke when.
    """
    with open(file_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            model="gpt-4o-transcribe-diarize",
            file=audio_file,
            response_format="diarized_json",
            chunking_strategy="auto",
        )

    segments = []
    for segment in result.segments:
        segments.append({
            "speaker": segment.speaker,
            "start": round(float(segment.start), 2),
            "end": round(float(segment.end), 2),
            "text": segment.text.strip(),
        })

    return {
        "full_text": result.text.strip(),
        "segments": segments,
    }
