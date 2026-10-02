# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **decision model**: a pretrained decoder LLM torso (Qwen3.5-2B-Base, a hybrid of Gated
DeltaNet and full-attention layers) with its LM head discarded and replaced by a ~1M-parameter
**pointer head**. It answers typed questions about a text (`state`) in one forward pass, with
no generation. The torso is adapted with rank-16 LoRA; the head runs in fp32. The reference
model is **v19** (`configs/train.yaml` + `training/recipe.sh`).

## Commands

```bash
pip install -e ".[dev]"          # add ".[dev,train]" for the corpus build / training
pytest -q                        # GPU tests skip without CUDA; suite downloads nothing from the Hub
pytest -q tests/test_core.py::test_confidence_uniform_is_zero   # a single test
pytest -q -m distributed         # torchrun multi-process tests on CPU (minutes); excluded by default, not run in CI
ruff check .                     # whole tree, as CI does
ruff format
mypy ./src                       # only the published package is type-checked (strict profile)
```

With uv: `uv run --extra dev pytest -q`, etc. Optional hatch shortcuts: `hatch run test`,
`hatch run lint:check` (ruff + mypy, exactly as the CI lint job runs it), `hatch run format`.

CI (`.github/workflows/tests.yml`) runs `pytest -q` with `HF_HUB_OFFLINE=1` on Python 3.10
(floor pins: torch 2.7.1, transformers 5.15.0, peft 0.21.0, huggingface_hub 1.5.0) and 3.12
(latest). Code must work on both. `pytest` puts `evaluation/` on `pythonpath`
(`test_hf_export` imports `multistep_eval`).

Running the model: `strands-decider ask <ckpt-or-hub-id> --state ... --noul/--choice/--score ...`
and `strands-decider serve <ckpt> --port 8000` (`POST /v1/systemone`, `/health`). Hidden
subcommands: `train`, `calibrate`, `eval`, `info`, plus `data build|stats|peek|recipes`.

## Conventions

- **Conventional Commits** are enforced (commitizen `commit-msg` hook and the PR-title gate in
  `.github/workflows/pr-title.yml`). Types: feat, fix, docs, style, refactor, perf, test,
  build, ci, chore, revert. Install hooks with `pre-commit install -t pre-commit -t commit-msg`.
- PRs target `main` (the GitHub default branch; CONTRIBUTING.md still names `staging`). PR and
  issue templates ask for a "Human Overview" section when an agent drafted the body.
- Keep mechanical changes (moves, renames, formatting) and functional changes in separate
  commits. Never change model inputs, calibration or training behaviour in a PR that claims to
  do something else.
- **No test checks the docs.** When moving a file or renaming a heading, grep all markdown and
  code comments for the old path/anchor and fix every reference. The docs cross-link heavily
  by heading anchor.
- **Changes to the recipe, data or model are experiments** and are preregistered: predictions
  and failure conditions committed in `research/preregistrations/` (with a config under
  `configs/experiments/`) before training; outcomes are appended afterwards, never edited.
  Files in `research/preregistrations/` are frozen and may name old paths — do not "fix" them
  (see the path map in that directory's README). Rank results on JevBench, not internal sets.
  Never modify the benchmark.
- Internal names: the project was formerly "hobson". Checkpoints saved before the rename
  (including the published v19) carry `hobson_config.json` instead of
  `strands_decider_config.json` (`LEGACY_CONFIG_NAME` in `modeling.py`); keep that fallback.

## Architecture (big picture)

Read `docs/architecture.md` for the full design; it ends with an annotated layout of every
module. The pieces that span several files:

- **Request → prompt → logits.** `schema.py` defines the wire types (`SystemOneRequest`,
  question types) and the confidence formulas. `prompting.py` renders state and questions into
  one prompt (`<state>…</state><question type=…>…<options>…</options></question><answer>`) and
  binds slots to options. `modeling.py` runs the torso and the readout: the query is the hidden
  state at `<answer>`, each key is the hidden state at the **last token of that option's
  line**; logits are their scaled dot product, then a masked softmax.
- **Three primitives, one mechanism.** `noul` (2 options → P(true)), `choice` (N options →
  argmax + confidence `(N·p_max−1)/(N−1)`), `score` (2–10 ordered levels → expected value,
  confidence from normalised std-dev with a floor corrected for ordinal smoothing). Per-kind
  temperatures from calibration are applied at inference.
- **No per-option parameters.** Labels come from the request, so the option count is limited
  only by the schema (255). Training depends on this: `data/collate.py` re-shuffles option
  order every example and remaps the label; score options are only ever reversed, never
  permuted. Don't introduce anything positional.
- **Drop, never truncate.** Training/eval drop rows that exceed the window. At serving time
  `SystemOneEngine._fit` in `infer.py` reserves the question tokens first and cuts the state
  from the front, so an option list is never cut.
- **Shared-prefix cache.** `infer.py` encodes the state once and forks the cache per question
  (`_expand_cache` / `_fork_layered_cache`). Gated DeltaNet layers hold conv/recurrent states
  that are updated in place, so the fork must copy into new layer objects; an unknown tensor
  raises `UnforkableCache` and the engine falls back to exact batched encoding. Many questions
  about one state are therefore nearly free.
- **Devices.** CUDA uses `flash-linear-attention` kernels; on macOS fla cannot run (no Triton build), so
  `mps_kernels.py` supplies the Gated DeltaNet chunk rule for MPS; CPU upcasts a bf16 torso to fp32 (faster there). CLI
  picks a device via `_auto_device` but docs recommend passing `--device` explicitly.
- **Training loss** (`train.py`, multi-GPU in `distributed.py` under torchrun): CE on gold
  (10% ordinal smoothing for scores) + KL to the frozen torso's own readout with the adapter
  disabled (0.3) + KL to stored frozen-teacher distributions on multi-step rows only (1.0,
  `data/teacher.py`).
- **Data pipeline** (`src/strands_decider/data/`): `recipes.py` turns HF datasets into
  `Example`s (`format.py`); `multistep.py`, `generated.py`, `adequacy.py`, `replay.py`,
  `catchall.py`, `distill.py` build the other corpora used by the recipe. `synth.py`,
  `synth_pairs.py`, `question_transforms.py`, `policy.py` are older experiments not used by
  the current recipe. `data/SHA256SUMS` pins committed and built files; the recipe checks it.
- **Export.** `hf_export.py` turns a checkpoint into a deterministic Hugging Face folder
  (safetensors, model card with provenance, `MANIFEST.sha256`); `verify` checks it.

## Training and GPU work

Training needs Linux/WSL2 with NVIDIA GPUs and the `train` extra plus pinned torch and
`flash-linear-attention` (`training/README.md#setup`). Entry points: `training/recipe.sh all`
(one host, `NGPU=1..8`) and `training/run_recipe.sh all` (multi-GPU host with per-stage
timing, row-count checks, S3 copy; needs `PY` and `S3_PREFIX`). **An agent operating training
or AWS hosts must follow `training/AGENTS.md`**: spend (launching hosts, capacity blocks) is the
user's decision, ship commits not working trees, one name tag/state dir per agent, tear down
and confirm `terminated`, and never change a script to make a check pass. Environment-specific
values (profile, bucket, regions) go in the git-ignored `training/aws/scripts/local.env`.
