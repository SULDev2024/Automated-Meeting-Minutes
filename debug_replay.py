"""
Targeted LLM replay for Meeting #2 (Steps 332-333).

Uses the transcript/diarization ALREADY SAVED in
exports/meeting-2-kazakh-final-v2.json. No audio, no transcription,
no diarization. Does not modify any pipeline code.

Usage (from the project folder, venv active):
    python debug_replay.py candidates   # Step 332: 1 LLM call
    python debug_replay.py resolve      # Step 333: resolution only
"""
import json
import sys

from backend.action_extractor import (
    build_indexed_transcript,
    find_action_candidates,
    attach_candidate_context,
    resolve_candidates_batched,
    resolve_candidates_with_context,
)

SOURCE = "exports/meeting-2-kazakh-final-v2.json"
CANDIDATES_OUT = "exports/debug-kazakh-candidates.json"
RESOLUTION_OUT = "exports/debug-kazakh-resolution.json"


def load_segments():
    with open(SOURCE, encoding="utf-8") as f:
        data = json.load(f)
    return data["diarization"]["segments"]


def load_meeting_language():
    # Same source production uses: the language detected during
    # transcription of the saved run.
    with open(SOURCE, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("transcription", {}).get("language", "Russian")


def show_candidates(segments, candidates):
    items = candidates.get("candidates", [])
    print(f"\nCandidate count: {len(items)}")
    for n, candidate in enumerate(items, 1):
        ids = candidate.get("segment_ids", [])
        print(f"\nCandidate {n}: {ids}")
        for segment_id in ids:
            try:
                index = int(segment_id.replace("SEG_", ""))
                print(f"   {segment_id} | {segments[index]['text']}")
            except (ValueError, IndexError, AttributeError):
                print(f"   {segment_id} | <invalid id>")


def run_candidates():
    segments = load_segments()
    transcript = build_indexed_transcript(segments)
    print("Calling find_action_candidates() once...")
    candidates = find_action_candidates(transcript)
    with open(CANDIDATES_OUT, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
    print(f"Saved raw output to {CANDIDATES_OUT}")
    show_candidates(segments, candidates)


def run_resolve(single_call=False):
    segments = load_segments()
    with open(CANDIDATES_OUT, encoding="utf-8") as f:
        candidates = json.load(f)
    enriched = attach_candidate_context(segments, candidates)
    count = len(enriched.get("candidates", []))
    language = load_meeting_language()
    print(f"Meeting language passed to resolver: {language}")
    if single_call:
        print(
            f"{count} candidates -> ONE call to "
            "resolve_candidates_with_context() (not production batching)."
        )
        resolved = resolve_candidates_with_context(
            enriched, meeting_language=language
        )
    else:
        print(
            f"{count} candidates -> production resolves them one per "
            f"call = {count} LLM call(s) (plus one retry per rate limit)."
        )
        resolved = resolve_candidates_batched(
            enriched, meeting_language=language
        )
    with open(RESOLUTION_OUT, "w", encoding="utf-8") as f:
        json.dump(resolved, f, ensure_ascii=False, indent=2)
    print(f"Saved raw output to {RESOLUTION_OUT}")
    results = resolved.get("results", [])
    actions = [r for r in results if r.get("is_action") is True]
    explicit = [r for r in results if r.get("is_action") in (True, False)]
    unresolved = [r for r in results if r.get("resolution_status") == "unresolved"]
    print(f"\nCandidate count: {count}")
    print(f"Resolved results: {len(results)} | is_action=True: {len(actions)}")
    print(f"Explicit decisions: {len(explicit)} | Unresolved: {len(unresolved)}")
    for n, r in enumerate(results, 1):
        print(
            f"\n{n} | is_action={r.get('is_action')} | owner={r.get('owner')}"
            f" | deadline={r.get('deadline')}"
            f"\n   task: {r.get('task')}"
            f"\n   candidate_ids: {r.get('candidate_segment_ids')}"
            f"\n   evidence_ids:  {r.get('evidence_segment_ids')}"
        )
        if r.get("resolution_error"):
            print(f"   resolution_error: {r.get('resolution_error')}")
    try:
        from backend.action_extractor import validate_candidate_resolution
        check = validate_candidate_resolution(enriched, resolved)
        print(
            "\nResolution validation:",
            "PASS" if check["valid"] else "FAIL",
            check["errors"],
        )
    except ImportError:
        pass


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "candidates":
        run_candidates()
    elif mode == "resolve":
        run_resolve()
    elif mode == "resolve-single":
        run_resolve(single_call=True)
    else:
        print("Usage: python debug_replay.py candidates | resolve | resolve-single")
