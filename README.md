# 🎙️ Automated Meeting Minutes

An AI-powered meeting assistant that transforms meeting recordings into structured, actionable meeting minutes.

The system combines **speech-to-text, speaker diarization, LLM-based analysis, action-item extraction, summarization, and document generation** to reduce the manual work required after meetings.

Built for **Russian, Kazakh and mixed Kazakh–Russian (Shala-Kazakh)** business meetings, where generic meeting tools struggle with code-switching, patronymics and inconsistent speech-to-text spelling of names.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-8-646CFF?logo=vite&logoColor=white)

---

## 📑 Table of Contents

- [The Problem](#-the-problem)
- [Our Solution](#-our-solution)
- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Output Format](#-output-format)
- [Evaluation](#-evaluation)
- [Known Limitations](#️-known-limitations)
- [Roadmap](#️-roadmap)

---

## 🎯 The Problem

Important information is often lost after meetings.

Teams may need to manually:

- Review long meeting recordings
- Identify who discussed what
- Find decisions and important points
- Extract assigned tasks
- Identify task owners and deadlines
- Write and format meeting minutes

This becomes especially difficult for long, multilingual, or informal conversations.

---

## 💡 Our Solution

**Automated Meeting Minutes** processes a meeting recording and converts it into useful structured information.

Instead of listening to the entire recording again, users can receive:

- 📝 Meeting transcription
- 👥 Speaker-separated conversation
- 📋 Meeting summary
- ✅ Action items
- 👤 Task owners
- 📅 Deadlines
- 📄 Structured meeting minutes
- 📥 Exportable PDF and DOCX documents

The goal is to turn an unstructured conversation into information that a team can immediately use.

---

## ✨ Key Features

### 🎙️ AI Speech Transcription
Converts meeting audio (`.mp3`, `.wav`, `.m4a`, `.mp4`, `.webm`) into text with automatic language detection (Whisper large-v3 via Groq).

### 👥 Speaker Diarization
Separates the conversation into speaker segments to preserve the structure of the meeting (OpenAI `gpt-4o-transcribe-diarize`).

### 🧠 AI Meeting Analysis
Uses large language models to understand the meeting and generate structured information.

### ✅ Action-Item Extraction
Detects tasks discussed during the meeting and extracts useful information such as:

- Task
- Owner
- Deadline

Beyond simple extraction, the pipeline provides:

- **Owner resolution from context.** For example, the chair addresses *"Nurlan Askarovich"* and gives the instruction two turns later.
- **Owner vs. target disambiguation.** In *"Timur, contact Nurlan"* the owner is Timur, not Nurlan.
- **Deadline detection.** This includes rule-based Kazakh weekday and date expressions, plus known speech-to-text spelling variants.
- **Evidence grounding.** Every action cites real transcript segment IDs (`SEG_005`, …), and evidence text is rebuilt from the transcript, never from LLM output.
- **De-duplication** of repeated assignments, instruction/commitment pairs, fragments and end-of-meeting recaps.
- **A validation report** that flags missing owners, missing evidence, unresolved speaker labels and mixed-script names.

### 📝 Automatic Summarization
Creates a concise executive summary of the most important meeting information: topics, decisions and risks. It is written in the meeting's own language.

### 🌐 Multilingual Meeting Processing
Designed to work with multilingual meeting scenarios, including experiments with **Kazakh and mixed-language/Shala-Kazakh speech**.

### 📄 Document Export
Generates professional meeting-minute documents that can be exported as:

- PDF
- DOCX

Full Cyrillic and Kazakh glyph support (`Ә Ғ Қ Ң Ө Ұ Ү Һ І`).

### 💻 Web Interface
Provides a frontend interface for interacting with the meeting-analysis system. It has drag-and-drop upload, a backend health indicator and an offline **demo mode** built from a saved real run.

---

## 🧠 How It Works

```text
Meeting Audio
     │
     ▼
┌─────────────────────┐
│ Speech Transcription│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Speaker Diarization │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   AI / LLM Analysis │
└──────────┬──────────┘
           │
     ┌─────┴─────────┐
     │               │
     ▼               ▼
 Meeting Summary   Action Items
                   Owner + Deadline
     │               │
     └───────┬───────┘
             ▼
    Structured Minutes
             │
       ┌─────┴─────┐
       ▼           ▼
      PDF         DOCX
```

### Action-item pipeline

`backend/action_extractor.py` → `extract_meeting_actions()` combines LLM reasoning with deterministic, rule-based post-processing:

| # | Stage | What it does |
|---|-------|--------------|
| 1 | Indexed transcript | Numbers every diarized segment as `SEG_000`, `SEG_001`, … |
| 2 | Candidate detection | LLM finds **every** passage that may contain a task, request, deadline or commitment (high recall). |
| 3 | Context attachment | Surrounding conversation is attached to each candidate. |
| 4 | Candidate resolution | Each candidate is resolved in **its own LLM call** (batching proved unreliable) into task / owner / evidence. |
| 5 | Addressee timeline | LLM detects who the chair is addressing and when, joining names split across segments. |
| 6 | Grounded actions | Evidence IDs are validated against the transcript; owners fall back to the addressee timeline; deadlines are found by rule-based matching. |
| 7 | Cleanup | Exact de-duplication, incomplete-fragment removal, owner normalization, speaker-label recovery, instruction ↔ commitment merging, contiguous-fragment merging. |
| 8 | Recap handling | Detects the end-of-meeting recap and removes repeats, recovering missing deadlines and linking short names (*"Даниар"*) to full names. |
| 9 | Language normalization | Rewrites tasks that drifted into another language back into the meeting language. |
| 10 | Validation | Produces a `valid` flag plus a list of issues for human review. |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | Python, FastAPI, Uvicorn |
| Transcription | Groq API: `whisper-large-v3` |
| Diarization | OpenAI API: `gpt-4o-transcribe-diarize` |
| Summary & reasoning | Groq API: `openai/gpt-oss-120b` (JSON mode, temperature 0) |
| Export | `python-docx`, ReportLab |
| Frontend | React 19, Vite 8, plain CSS |

---

## 📁 Project Structure

```
Automated-Meeting-Minutes/
├── backend/
│   ├── main.py               # FastAPI app and endpoints
│   ├── transcriber.py        # Groq Whisper transcription + language detection
│   ├── diarization.py        # OpenAI diarized transcription
│   ├── summarizer.py         # Executive summary generation
│   ├── action_extractor.py   # Action-item extraction, owner/deadline resolution, de-duplication
│   └── exporter.py           # DOCX and PDF minutes generation (Unicode font discovery)
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Upload, results and demo UI
│   │   ├── App.css
│   │   └── data/demoMeeting.json   # Saved real run used by "Load Demo"
│   ├── public/demo/          # Demo PDF / DOCX exports
│   └── vite.config.js        # Proxies /api → http://127.0.0.1:8000
├── audio-sample/             # Russian, Kazakh and Shala-Kazakh test recordings + script
├── evaluation/               # Ground truth and scored evaluation runs
├── debug_replay.py           # Re-runs LLM stages on a saved result without re-transcribing
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Node.js 20+** and npm
- A **Groq API key**: <https://console.groq.com>
- An **OpenAI API key** with access to `gpt-4o-transcribe-diarize`: <https://platform.openai.com>
- *(Linux only, for PDF export)* a Unicode TTF font such as DejaVu Sans:
  `sudo apt install fonts-dejavu-core`

### 1. Clone the repository

```bash
git clone https://github.com/SULDev2024/Automated-Meeting-Minutes.git
cd Automated-Meeting-Minutes
```

### 2. Backend setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```

> Both keys are required, and the backend refuses to start if either is missing. `.env` is already in `.gitignore`; never commit it.

Start the API **from the project root**, because it writes uploads to `data/` and exports to `exports/` relative to it:

```bash
uvicorn backend.main:app --reload
```

The API runs at <http://127.0.0.1:8000>, with interactive docs at <http://127.0.0.1:8000/docs>.

### 3. Frontend setup

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The Vite dev server proxies `/api/*` to the backend on port 8000.

---

## 📖 Usage

1. Check that the header shows **System Online** (the backend is reachable).
2. Drag a recording onto the upload area, or click to browse.
3. Click **Analyze Meeting**. Processing time depends on recording length, since each action candidate is resolved with its own LLM call.
4. Review the summary, action items (owner + deadline) and full transcript.
5. Click **Download PDF** or **Download DOCX** for the minutes.

**No API keys?** Click **Load Demo** to view a saved real analysis of `meeting-2-kazakh.mp3`. No backend or AI call is needed.

Sample recordings to try are in [`audio-sample/`](audio-sample/).

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Service info |
| `GET`  | `/health` | Health check → `{"status": "ok"}` |
| `POST` | `/meetings/analyze` | **Full pipeline**: transcription, diarization, summary, action items and exports |
| `GET`  | `/meetings/{meeting_id}/download/pdf` | Download generated PDF minutes |
| `GET`  | `/meetings/{meeting_id}/download/docx` | Download generated DOCX minutes |
| `POST` | `/transcribe` | Transcription only |
| `POST` | `/diarize` | Diarization only |

All `POST` endpoints accept `multipart/form-data` with a single `file` field.

```bash
curl -X POST http://127.0.0.1:8000/meetings/analyze \
  -F "file=@audio-sample/meeting-2-kazakh.mp3"
```

---

## 📦 Output Format

`POST /meetings/analyze` returns (abridged):

```json
{
  "status": "success",
  "meeting_id": "ad72eb4b213a43149c9344a94d5ea14b",
  "filename": "meeting-2-kazakh.mp3",
  "transcription": { "language": "Kazakh", "segments": [...], "full_text": "..." },
  "diarization":   { "segments": [{ "speaker": "A", "start": 0.0, "end": 1.36, "text": "..." }], "full_text": "..." },
  "summary": "Executive summary in the meeting language...",
  "addressee_timeline": { ... },
  "action_items": [
    {
      "task": "Collect the missing supplier documents and send the results",
      "owner": "Нұрлан Асқарулы",
      "deadline": "жұмаға дейін",
      "evidence_segment_ids": ["SEG_005", "SEG_006", "SEG_007"],
      "evidence": "SEG_005 A: Нұрлан Асқарулы,\nSEG_006 A: ..."
    }
  ],
  "action_validation": { "valid": true, "issues": [], "resolution": { ... } },
  "exports": {
    "docx": "/meetings/ad72eb4b.../download/docx",
    "pdf":  "/meetings/ad72eb4b.../download/pdf"
  }
}
```

If analysis succeeds but export fails, the result is still returned with `"exports": null` and an `export_error` message.

---

## 📊 Evaluation

The [`evaluation/`](evaluation/) folder holds ground truth and scored runs. A scripted Shala-Kazakh meeting ([`audio-sample/shala-kazakh-test-script.txt`](audio-sample/shala-kazakh-test-script.txt)) with **4 known assignments** was recorded and processed end-to-end.

**Shala-Kazakh Run 01** (`meeting-3-shala-kazakh.mp3`):

| Metric | Result |
|--------|--------|
| Semantic task coverage | **100%** (4 / 4) |
| Owner accuracy | **100%** |
| Structured deadline accuracy | **75%** (3 / 4) |
| Detected action records | 12 (over-generation ×3.0) |
| Duplicate / fragment records | 7 |

Every intended assignment and owner was found. The main weakness on heavily code-switched speech is **fragmentation**: one assignment spread across several records. The recap and fragment-merging stages in the pipeline were added to address this.

---

## ⚠️ Known Limitations

- **Cloud-based processing.** Audio and transcripts are sent to Groq and OpenAI. Don't use it for confidential meetings without checking those providers' data policies.
- **Two APIs, two passes.** Audio is transcribed twice: once for language detection and once for diarization.
- **Speaker labels are anonymous** (`A`, `B`, `C`…). Real names are inferred from how people address each other.
- **Latency and cost** grow with meeting length, because each action candidate takes a separate LLM call.
- **Synchronous processing.** `/meetings/analyze` blocks until the whole pipeline finishes, and concurrent uploads share one temporary file path.
- **Exports are stored on local disk** in `exports/`, with no expiry or authentication.
- **Tuned for Russian/Kazakh.** Deadline rules are strongest for Kazakh expressions; other languages rely on the LLM alone.

---

## 🗺️ Roadmap

- [ ] Background job queue with progress reporting for long recordings
- [ ] Reuse one transcription for both language detection and diarization
- [ ] Automated evaluation script that scores runs against ground truth
- [ ] Editable action items in the UI before export
- [ ] Docker / docker-compose setup
- [ ] Unit tests for the rule-based deadline and de-duplication stages
