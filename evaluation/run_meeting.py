"""
Run one evaluation meeting and save the complete output.

Transcript sources:
    audio   transcribe + diarize the recording (production path)
    oracle  use the scripted turns from the gold file (no ASR errors)
    run     reuse the transcript saved in an earlier run (--from-run)

Extraction methods:
    pipeline     the full production action-item pipeline
    pipeline_v1  the pipeline before the multi-result fix (ca3a9ab), for comparison
    baseline     a single LLM call that extracts all action items at once

Examples (from the project root, venv active):
    python -m evaluation.run_meeting --gold G/S1-ru.json --source audio --audio-dir REC/
    python -m evaluation.run_meeting --gold G/S1-ru.json --source oracle
    python -m evaluation.run_meeting --gold G/S1-ru.json --source run \
        --from-run evaluation/runs/S1-ru__audio__pipeline__r1.json --method baseline
"""
import argparse
import json
import os
import subprocess
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

MODELS = {
    "transcription": "groq:whisper-large-v3",
    "diarization": "openai:gpt-4o-transcribe-diarize",
    "llm": "groq:openai/gpt-oss-120b",
}
LLM_MODEL = "openai/gpt-oss-120b"
USAGE = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def count_usage(client):
    """Wrap a Groq client so every chat call adds its token usage to USAGE."""
    create = client.chat.completions.create

    def counted(*args, **kwargs):
        response = create(*args, **kwargs)
        USAGE["llm_calls"] += 1
        if getattr(response, "usage", None):
            USAGE["prompt_tokens"] += response.usage.prompt_tokens or 0
            USAGE["completion_tokens"] += response.usage.completion_tokens or 0
        return response

    client.chat.completions.create = counted
    return client


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def oracle_segments(gold: dict) -> list:
    # Fake one-second timestamps; speaker labels stay anonymous (A-D),
    # exactly like diarization output, so no names leak from the labels.
    return [
        {
            "speaker": turn["speaker"],
            "start": float(index),
            "end": float(index + 1),
            "text": turn["text"],
        }
        for index, turn in enumerate(gold["turns"])
    ]


def find_recording(audio_dir: str, meeting_id: str) -> str:
    # Recordings may come from phones or meeting apps in any supported format.
    for extension in (".mp3", ".m4a", ".wav", ".webm", ".mp4"):
        path = os.path.join(audio_dir, meeting_id + extension)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No recording for {meeting_id} in {audio_dir}")


def transcribe_recording(path: str):
    from backend.transcriber import transcribe_audio
    from backend.diarization import diarize_audio

    transcription = transcribe_audio(path)
    diarization = diarize_audio(path)
    return transcription, diarization


def run_baseline(segments: list, meeting_language: str) -> dict:
    """Single-prompt extraction: the approach most meeting tools use."""
    from groq import Groq
    from backend.action_extractor import build_indexed_transcript

    transcript = build_indexed_transcript(segments)
    prompt = f"""Extract all action items from this meeting transcript.

TRANSCRIPT:
{transcript}

For every action item return the task, the responsible person (owner),
the deadline exactly as stated (or null), and the IDs of the transcript
segments that support it. Write the task in {meeting_language}.

Return valid JSON only:
{{
  "action_items": [
    {{
      "task": "...",
      "owner": "... or null",
      "deadline": "... or null",
      "evidence_segment_ids": ["SEG_000"]
    }}
  ]
}}
"""
    client = count_usage(Groq(api_key=os.getenv("GROQ_API_KEY"), max_retries=8))
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    items = json.loads(response.choices[0].message.content).get("action_items", [])
    return {"action_items": items, "validation": None}


def run_pipeline(segments: list, meeting_language: str, v1: bool = False) -> dict:
    import backend.action_extractor as extractor

    if v1:
        # v1 behaviour: a candidate answered with several actions is
        # rejected as a whole (before commit ca3a9ab).
        extractor._accept_multiple_results = lambda candidate, matching: [
            extractor.unresolved_candidate_result(
                candidate,
                f"Expected exactly one result for the sent candidate, got {len(matching)}",
            )
        ]

    # Evaluation runs hit rate limits and network drops far more often than
    # single uploads; let the SDK retry with backoff instead of failing.
    extractor.client = count_usage(extractor.client.with_options(max_retries=8))
    result = extractor.extract_meeting_actions(segments, meeting_language=meeting_language)
    # The pipeline records a failed LLM call as an unresolved candidate and
    # carries on. Such a run measures the API, not the method: reject it.
    if "LLM call failed" in json.dumps(result.get("validation"), ensure_ascii=False):
        raise RuntimeError("An LLM call failed inside the pipeline; run discarded")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--gold", required=True, help="gold JSON of the meeting")
    parser.add_argument("--source", choices=["audio", "oracle", "run"], required=True)
    parser.add_argument("--method", choices=["pipeline", "pipeline_v1", "baseline"], default="pipeline")
    parser.add_argument("--audio-dir", help="folder with <meeting_id>.mp3 (source=audio)")
    parser.add_argument("--from-run", help="earlier run JSON to reuse (source=run)")
    parser.add_argument("--tag", default="r1", help="repetition label, e.g. r1, r2, r3")
    parser.add_argument("--out", default="evaluation/runs")
    args = parser.parse_args()

    with open(args.gold, encoding="utf-8") as f:
        gold = json.load(f)
    meeting_id = gold["meeting_id"]
    meeting_language = gold["pipeline_language"]
    transcription = None

    if args.source == "audio":
        if not args.audio_dir:
            parser.error("--audio-dir is required for --source audio")
        audio_path = find_recording(args.audio_dir, meeting_id)
        transcription, diarization = transcribe_recording(audio_path)
        segments = diarization["segments"]
        # Production uses the language Whisper detects, so we do too.
        meeting_language = transcription.get("language") or meeting_language
        transcript_source = audio_path
    elif args.source == "oracle":
        segments = oracle_segments(gold)
        transcript_source = "gold turns"
    else:
        if not args.from_run:
            parser.error("--from-run is required for --source run")
        with open(args.from_run, encoding="utf-8") as f:
            previous = json.load(f)
        segments = previous["segments"]
        transcription = previous.get("transcription")
        meeting_language = previous["run_info"]["meeting_language"]
        transcript_source = args.from_run

    if args.method == "baseline":
        result = run_baseline(segments, meeting_language)
    else:
        result = run_pipeline(segments, meeting_language, v1=args.method == "pipeline_v1")

    # A reused transcript is still an ASR transcript; name the run by its origin.
    source_name = "audio" if args.source == "run" else args.source
    run_name = f"{meeting_id}__{source_name}__{args.method}__{args.tag}"
    output = {
        "run_info": {
            "run_name": run_name,
            "meeting_id": meeting_id,
            "source": source_name,
            "transcript_source": transcript_source,
            "method": args.method,
            "tag": args.tag,
            "meeting_language": meeting_language,
            "models": MODELS,
            "git_commit": git_commit(),
            "llm_usage": USAGE,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "transcription": transcription,
        "segments": segments,
        "action_items": result.get("action_items", []),
        "validation": result.get("validation"),
        "addressee_timeline": result.get("addressee_timeline"),
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{run_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"{run_name}: {len(output['action_items'])} action items -> {path}")


if __name__ == "__main__":
    main()
