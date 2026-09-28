# 🎙️ Automated Meeting Minutes

An AI-powered meeting assistant that transforms meeting recordings into structured, actionable meeting minutes.

The system combines **speech-to-text, speaker diarization, LLM-based analysis, action-item extraction, summarization, and document generation** to reduce the manual work required after meetings.

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
Converts meeting audio into text for further analysis.

### 👥 Speaker Diarization
Separates the conversation into speaker segments to preserve the structure of the meeting.

### 🧠 AI Meeting Analysis
Uses large language models to understand the meeting and generate structured information.

### ✅ Action-Item Extraction
Detects tasks discussed during the meeting and extracts useful information such as:

- Task
- Owner
- Deadline

### 📝 Automatic Summarization
Creates a concise summary of the most important meeting information.

### 🌐 Multilingual Meeting Processing
Designed to work with multilingual meeting scenarios, including experiments with **Kazakh and mixed-language/Shala-Kazakh speech**.

### 📄 Document Export
Generates professional meeting-minute documents that can be exported as:

- PDF
- DOCX

### 💻 Web Interface
Provides a frontend interface for interacting with the meeting-analysis system.

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