# Evaluation harness

Runs the pipeline on evaluation meetings, saves every run completely, and scores the results against ground truth.

## Ground truth format

One JSON file per meeting (`<meeting_id>.json`) with:
- `meeting_id`, `language` (`ru` / `kk` / `mix`), `pipeline_language`, `audio` (file name of the recording)
- `speakers`: label (`A`–`D`) → `name` and an `owner_pattern` regex, matched after lower-casing and mapping Kazakh letters to Russian ones (`ә→а`, `қ→к`, …)
- `turns`: the scripted dialogue (used by the oracle source)
- `tasks`: `id`, `owner` (speaker label), `task_en`, `deadline_en`, `deadline_patterns` (all must match; `null` = no deadline), `hard_case`

## 1. Run meetings

From the project root, with the venv active and `.env` configured:

```bash
G=path/to/gold          # ground-truth folder
REC=path/to/recordings  # <meeting_id>.mp3 / .m4a / .wav files

for g in $G/*.json; do
  id=$(basename "$g" .json)
  for r in r1 r2 r3; do
    # Full production path: ASR + diarization + pipeline
    python -m evaluation.run_meeting --gold "$g" --source audio --audio-dir "$REC" --tag $r
    # Baseline on the SAME transcript (no second ASR call)
    python -m evaluation.run_meeting --gold "$g" --source run --method baseline --tag $r \
      --from-run "evaluation/runs/${id}__audio__pipeline__${r}.json"
  done
  # Oracle transcript (no ASR errors): isolates the effect of ASR
  python -m evaluation.run_meeting --gold "$g" --source oracle --tag r1
done
```

Each run is saved to `evaluation/runs/<meeting>__<source>__<method>__<tag>.json` with the transcript, action items, validation output, models, git commit and timestamp.

## 2. Score

```bash
python -m evaluation.score --gold-dir $G --runs-dir evaluation/runs
```

1. **Alignment.** An LLM judge links each predicted action item to at most one ground-truth task, comparing only *what* has to be done. The result is saved as `<run>.alignment.json`. **Check these by hand** and set `"reviewed": true`. Saved alignments are always reused, so your corrections are never overwritten. Use `--judge manual` for empty templates instead.
2. **Scoring** is deterministic. Owner and deadline are checked with the regex patterns from the ground truth.

| Metric | Definition |
|---|---|
| Task recall | gold tasks matched by ≥ 1 prediction / gold tasks |
| Precision | gold tasks matched / predicted records (one-to-one) |
| Owner / deadline accuracy | correct among matched tasks |
| Strict accuracy | task + owner + deadline all correct / gold tasks |
| Over-generation | predicted records / gold tasks |
| Duplicates / spurious | extra matches to an already-matched task / predictions matching nothing |

When several predictions match one task, the best-scoring one is used and the rest count as duplicates.

Output: `evaluation/runs/results/summary.md`, `summary.csv`, `per_run.csv`, `per_task.csv`. `per_task.csv` includes each task's `hard_case` label for error analysis.

Run outputs contain meeting transcripts and are git-ignored.
