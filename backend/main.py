from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from backend.transcriber import transcribe_audio
from backend.diarization import diarize_audio
from backend.summarizer import summarize_meeting
from backend.action_extractor import extract_meeting_actions
from backend.exporter import export_meeting_docx, export_meeting_pdf

import os
import re
import shutil
import tempfile
import uuid


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORTS_DIR = os.path.join(PROJECT_ROOT, "exports")
MEETING_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
EXPORT_FORMATS = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pdf": "application/pdf",
}


def export_filename(meeting_id: str, export_format: str) -> str:
    return f"meeting_{meeting_id}_minutes.{export_format}"


def export_path(meeting_id: str, export_format: str) -> str:
    return os.path.join(EXPORTS_DIR, export_filename(meeting_id, export_format))


def generate_meeting_exports(meeting_id: str, meeting_result: dict) -> dict:
    """Write DOCX + PDF for a finished meeting; return download URLs only."""
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    export_meeting_docx(meeting_result, export_path(meeting_id, "docx"))
    export_meeting_pdf(meeting_result, export_path(meeting_id, "pdf"))
    return {
        export_format: f"/meetings/{meeting_id}/download/{export_format}"
        for export_format in ("docx", "pdf")
    }


app = FastAPI(
    title="Automated Meeting Minutes",
    description="Local AI system for meeting transcription and action-item tracking",
    version="0.1.0"
)


def build_meeting_result(
    diarization: dict,
    summary: str,
    action_items: list
):
    return {
        "status": "success",
        "transcript": {
            "full_text": diarization.get(
                "full_text",
                ""
            ),
            "segments": diarization.get(
                "segments",
                []
            ),
        },
        "summary": summary,
        "action_items": action_items,
        "statistics": {
            "speaker_labels": len(
                set(
                    segment.get("speaker")
                    for segment in diarization.get(
                        "segments",
                        []
                    )
                    if segment.get("speaker")
                )
            ),
            "action_items": len(action_items),
        },
    }


@app.get("/")
def root():
    return {
        "project": "Automated Meeting Minutes",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/meetings/analyze")
async def analyze_meeting(file: UploadFile = File(...)):
    allowed_extensions = (
        ".wav",
        ".mp3",
        ".m4a",
        ".mp4",
        ".webm",
    )
    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type"
        )
    temp_path = os.path.join(
        "data",
        f"meeting_upload{extension}"
    )
    os.makedirs("data", exist_ok=True)
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        transcription = transcribe_audio(temp_path)
        diarization = diarize_audio(temp_path)
        detected_language = transcription.get(
            "language",
            "Russian"
        )
        summary = summarize_meeting(
            diarization.get("full_text", ""),
            language=detected_language
        )
        action_result = extract_meeting_actions(
            diarization.get("segments", []),
            meeting_language=detected_language
        )
        action_items = action_result.get(
            "action_items",
            []
        )
        addressee_timeline = action_result.get("addressee_timeline", {})
        action_validation = action_result.get(
            "validation",
            {
                "valid": False,
                "issues": [
                    "Action validation unavailable"
                ]
            }
        )
        meeting_id = uuid.uuid4().hex
        result = {
            "status": "success",
            "meeting_id": meeting_id,
            "filename": file.filename,
            "transcription": transcription,
            "diarization": diarization,
            "summary": summary,
            "addressee_timeline": addressee_timeline,
            "action_items": action_items,
            "action_validation": action_validation,
        }
        # Analysis already succeeded; an export failure must not discard it.
        try:
            result["exports"] = generate_meeting_exports(meeting_id, result)
        except Exception as export_error:
            result["exports"] = None
            result["export_error"] = str(export_error)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def download_export(meeting_id: str, export_format: str):
    if not MEETING_ID_PATTERN.match(meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")
    path = export_path(meeting_id, export_format)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Export not found")
    return FileResponse(
        path,
        media_type=EXPORT_FORMATS[export_format],
        filename=export_filename(meeting_id, export_format),
    )


@app.get("/meetings/{meeting_id}/download/docx")
def download_docx(meeting_id: str):
    return download_export(meeting_id, "docx")


@app.get("/meetings/{meeting_id}/download/pdf")
def download_pdf(meeting_id: str):
    return download_export(meeting_id, "pdf")


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):

    allowed_extensions = {
        ".wav",
        ".mp3",
        ".m4a",
        ".mp4",
        ".webm"
    }

    extension = os.path.splitext(file.filename)[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type."
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        result = transcribe_audio(temp_path)

        return {
            "status": "success",
            "filename": file.filename,
            "transcription": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/diarize")
async def diarize(file: UploadFile = File(...)):
    allowed_extensions = {".wav", ".mp3", ".m4a", ".mp4", ".webm"}
    extension = os.path.splitext(file.filename)[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type."
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        result = diarize_audio(temp_path)

        return {
            "status": "success",
            "filename": file.filename,
            "diarization": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
