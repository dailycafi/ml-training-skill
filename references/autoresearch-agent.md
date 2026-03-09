# Autonomous Experiment Agent (autoresearch pattern)

Guide for setting up an AI agent to run autonomous ML experiments using the
autoresearch keep/discard/revert loop. Read this when you want Claude or another
LLM agent to iterate on a training script overnight without human intervention.

For the training patterns themselves, see `experiment-loop.md`.

## Project structure

The autoresearch pattern separates concerns into two files with strict boundaries:

### `prepare.py` — Fixed infrastructure (read-only during experiments)

Contains everything that should NOT change between experiments:

| Responsibility | Details |
|---|---|
| **Constants** | `MAX_SEQ_LEN`, `TIME_BUDGET`, `EVAL_TOKENS`, `VOCAB_SIZE` |
| **Data download** | Parallel shard download with retries |
| **Tokenizer training** | BPE via rustbpe → tiktoken pickle + `token_bytes.pt` for BPB eval |
| **Tokenizer class** | Wraps tiktoken with `encode()`, `decode()`, BOS handling |
| **Dataloader** | BOS-aligned packing with best-fit bin packing, pinned memory, async GPU transfer |
| **Evaluation** | `evaluate_bpb()` — the ground truth metric, never modified |

Key design choices:
- **Pinned validation shard**: last shard is always val, deterministic across experiments
- **BPB (bits per byte)**: vocab-size-independent metric, enables fair comparison when changing tokenizer
- **Best-fit document packing**: 100% token utilization (no padding), each row starts with BOS

### `train.py` — The only file agents edit

Contains everything that IS fair game for experimentation:

| Responsibility | Details |
|---|---|
| **Model architecture** | Full GPT definition (attention, MLP, embeddings, RoPE) |
| **Optimizer** | MuonAdamW with per-group config (matrix params → Muon, rest → AdamW) |
| **Hyperparameters** | All tunable knobs as top-level constants (no CLI flags) |
| **Training loop** | Forward/backward, gradient accumulation, LR scheduling, logging |
| **Output format** | Prints `val_bpb`, `peak_vram_mb`, `mfu_percent`, etc. in parseable format |

### Imports between files

```python
# train.py imports fixed infrastructure from prepare.py
from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb
```

This boundary ensures experiments only vary the model/optimizer while data and evaluation remain constant.

## Agent instructions template (program.md)

The `program.md` file guides the AI agent's behavior. Adapt this template to your project:

```markdown
# <project-name>

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`).
   The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from main.
3. **Read the in-scope files**: Read these files for full context:
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that data/tokenizer are prepared. If not, run `prepare.py`.
5. **Initialize results.tsv**: Create with header row only. Baseline recorded after first run.
6. **Confirm and go**: Confirm setup looks good.

## Experimentation

Each experiment runs on a single GPU with a **fixed time budget of N minutes**.
Launch: `uv run train.py > run.log 2>&1`

**What you CAN do:**
- Modify `train.py` — architecture, optimizer, hyperparameters, batch size, model size.

**What you CANNOT do:**
- Modify `prepare.py` (read-only: evaluation, data loading, tokenizer, constants).
- Install new packages or add dependencies.
- Modify the evaluation harness.

**The goal: get the lowest val_bpb.**

**Simplicity criterion**: All else being equal, simpler is better. A 0.001 improvement
that adds 20 lines of hacky code? Probably not worth it. A 0.001 improvement from
deleting code? Definitely keep. Equal metric but simpler code? Keep.

**The first run**: Always establish the baseline by running train.py as-is.

## The experiment loop

(See `experiment-loop.md` for the full loop steps and keep/discard/revert rules.)

Run each experiment: `uv run train.py > run.log 2>&1`
Parse results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
If improved → KEEP. If worse → DISCARD (`git reset --hard HEAD~1`).

**NEVER STOP**: Do not pause to ask the human. Run indefinitely until manually stopped.
If stuck, think harder — re-read code, try combining near-misses, try radical changes.
```

## Key agent behavior rules

- **Redirect stdout**: `> run.log 2>&1` prevents output from flooding the agent's context window
- **Parse results via grep**: `grep "^val_bpb:" run.log` extracts metrics without reading full logs
- **Never stop**: the agent runs autonomously — no "should I continue?" prompts
- **One change at a time**: isolates the effect of each modification
- **Crash triage**: trivial bugs (typo, import) → fix and retry; fundamental (OOM) → log and move on
- **10-minute timeout**: if a run exceeds 2× budget, kill it and treat as failure
