# Translation_Model

Comparing different model translation ability and fine tune the model for special language.

**subtune** fine-tunes text-to-text translation models (TranslateGemma via GGUF, Hy-MT2, Hunyuan-MT,
Seed-X, Qwen3, or any public Hugging Face chat model or GGUF) on parallel movie and TV subtitles from
OpenSubtitles, through a Gradio web app. Every model it offers downloads without a token or an
access request.

## Quick start (Docker)

```bash
cp .env.example .env
```

Fill in `.env`, then:

```bash
docker compose up -d --build
```

Open http://localhost:7860. Downloaded subtitles, the catalog and converted GGUF models are kept in
`./data`, fine-tuned models in `./outputs`, and Hugging Face downloads in the `hf-cache` volume.

The image uses PyTorch for ROCm 7.2 and passes the AMD GPU (`/dev/kfd`, `/dev/dri`) into the
container; the host needs the `amdgpu` kernel driver. To use another PyTorch build, set
`TORCH_INDEX_URL` (for example `https://download.pytorch.org/whl/rocm7.1`) in `.env` before
building. On a machine without an AMD GPU (training on Modal only):

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu docker compose -f docker-compose.yml -f docker-compose.no-gpu.yml up -d --build
```

Files the container writes to `./data` and `./outputs` are owned by root.

## What it does

1. **Data:** pick a source and target language (for example English → Chinese (Traditional)). The
   app lists every movie and TV series that has subtitles in both languages, with the total count,
   and lets you select titles with a "top N by popularity" slider and per-row checkboxes.
2. **Download and align:** for each selected title it downloads the best subtitle in each language
   (the most downloaded human translation) and aligns the two by timestamp. Each file is downloaded
   once and kept, so later runs on other models cost no quota.
3. **Train:** LoRA, QLoRA or full fine-tuning, on the local AMD R9700 (ROCm) or a Modal cloud GPU.
   The base model and the fine-tuned model are both scored (chrF and BLEU) on held-out titles.
4. **Tag:** the result is saved as `outputs/models/<model>__<src>-<tgt>__<method>__<date>/` with a
   `metadata.json` and a model card. The Models tab lists every tag with its scores.
5. **GGUF export (optional):** the fine-tuned model is also written as a `.gguf` file (Q4_K_M by
   default) for llama.cpp, Ollama and LM Studio.

## Setup without Docker

```bash
uv venv && source .venv/bin/activate
```

Install PyTorch for your GPU first. For the AMD R9700, use the ROCm wheel that matches your installed
ROCm version (see https://pytorch.org/get-started/locally/):

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/rocmX.Y
```

Then install the app with its training and cloud extras:

```bash
uv pip install -e ".[train,qlora,cloud,dev]"
```

Copy the env file and fill it in:

```bash
cp .env.example .env
```

| Variable | What for |
|---|---|
| `OPENSUBTITLES_API_KEY` | Your API consumer key (opensubtitles.com → API consumers). |
| `OPENSUBTITLES_USERNAME`, `OPENSUBTITLES_PASSWORD` | Your account; logging in gives the VIP download quota. |
| `OPENSUBTITLES_USER_AGENT` | Name of your app, e.g. `subtune v0.1`. |
| `OPENSUBTITLES_RATE_LIMIT` | Requests per second (5 by default; 50 only on a paid API plan). |
| `HF_TOKEN` | Optional. All offered models are ungated; a token only raises download rate limits. |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | For cloud training (modal.com → Settings → API tokens). |

## Run

```bash
subtune ui
```

Open http://127.0.0.1:7860.

The same steps are available on the command line:

```bash
subtune scan --src en --tgt zh-tw
```

```bash
subtune quota
```

```bash
subtune check-qlora
```

## How the movie list is built

The OpenSubtitles API can't search for "has subtitles in both A and B". So the app pages through
every subtitle in the rarer language (free, no quota), then checks each of those titles for the
other language (one search per title). Both steps are cached and resume where they stopped. Expect
one search per title at the default 5 requests/second the first time; the scan of a language is
reused by every pair that includes it. Tick **Rescan for new subtitles** to pick up new uploads.

## Quota

Only downloads use quota: 2 per movie or episode (one file per language); VIP gives 1000 per day.
If a selection needs more downloads than you have left, **Start training** tells you how many
titles are ready and lets you either train now on the ready ones (the tag records the real count)
or download only and continue the next day.

## Models

All of these are public and ungated on Hugging Face (no token, no access request):

| Model | Notes |
|---|---|
| TranslateGemma 4B / 12B / 27B (GGUF) | Ungated GGUF copies by mradermacher (f16 for 4B, Q8_0 for 12B/27B). Gemma Terms of Use apply. |
| Hy-MT2 7B | Apache-2.0. Tencent's translation model; uses its documented prompts. |
| Hunyuan-MT 7B | Tencent Hunyuan Community License (check its territory restrictions). |
| Seed-X PPO 7B | OpenMDW license. Raw prompt ending in a target-language tag. |
| Qwen3 4B / 8B / 14B | Apache-2.0. Thinking is turned off in the prompt. |
| Custom | Any public causal LM or GGUF repo. The app checks Hugging Face and refuses gated or private models. |

Google's own TranslateGemma repos are gated (they require an access request), so the app uses
community GGUF copies of the same weights, which download freely.

### GGUF models

GGUF is the llama.cpp file format. Transformers can't train a quantized GGUF directly, so on first
use the app converts the chosen file into a regular bf16 checkpoint (cached in
`data/gguf-converted/`, or on the Modal cache volume) and trains from that. The conversion needs
RAM for the whole model: about 16 GB for 4B, 40 GB for 12B and 80 GB for 27B. On Modal the app
reserves that RAM for GGUF runs.

For a custom GGUF repo, enter its id (for example `bullerwins/translategemma-4b-it-GGUF`) and pick a
file. Prefer f16, bf16 or Q8_0 files: lower-bit quantization errors carry into the fine-tune.
Transformers reads GGUF for the common architectures (Llama, Mistral, Qwen2/3, Gemma 2/3 and
others); Hunyuan GGUFs are not supported, so use the regular Hy-MT2 and Hunyuan-MT repos.

Training-memory estimates are shown in the app. On the 32 GB R9700, LoRA fits the 4B–12B models;
Qwen3 14B and TranslateGemma 27B need QLoRA or a Modal GPU. QLoRA on ROCm depends on bitsandbytes support for the
card; the app checks it before starting and suggests Modal if it doesn't work.

## Exporting to GGUF

Pick a type under **Also export as GGUF** on the Train tab: Q4_K_M (the usual choice for Ollama and
LM Studio), Q8_0, F16 or BF16. After training, the file is written to
`outputs/models/<tag>/gguf/<tag>.<type>.gguf`, and the Models tab shows it.

- Base model from a regular Hugging Face repo: the LoRA adapter is merged into the bf16 base model,
  then llama.cpp's `convert_hf_to_gguf.py` writes the GGUF with the chat template embedded.
- Base model from a GGUF (TranslateGemma): the adapter is converted with `convert_lora_to_gguf.py`
  and merged into the *original* GGUF with `llama-export-lora`, so the file keeps the original
  tokenizer exactly. A full fine-tune of a GGUF base can't be exported; use LoRA or QLoRA.

The merge needs RAM for the whole bf16 model (about 2 GB per billion parameters); on Modal the app
reserves it. llama.cpp is pinned to release `b11989`. The Docker and Modal images include it
prebuilt; without Docker, the app clones and builds it on first use (needs git, cmake and a C++
compiler, a few minutes). If an export fails, the fine-tuned model is still saved and the error is
shown in the Models tab.

## Cloud costs (Modal)

The GPU is billed per second while a job runs. A job is shut down:

- when training finishes or fails: the job runs as an ephemeral Modal app, which stops as soon as
  the launcher exits; an idle container is released after Modal's default 60-second window;
- when you press **Stop training** (the launcher exits cleanly, which stops the Modal app);
- if the launcher dies or loses its connection (Modal stops the app when the client disconnects);
- at the latest after **Max cloud hours** (default 8), a hard limit Modal enforces on its side.

Failed jobs are never retried. After the weights are downloaded, the run's files are deleted from
the `subtune` Modal volume; the `subtune-hf-cache` volume keeps downloaded base models so the next
run starts faster (volumes include 1 TiB free). To cap spending on top of this, set a budget in
the Modal dashboard.

## Layout

```
data/catalog.sqlite      what OpenSubtitles has, what is downloaded and aligned
data/subtitles/<lang>/   downloaded .srt files, one per OpenSubtitles file id
data/pairs/<src>__<tgt>/ aligned sentence pairs per movie/episode (JSONL)
data/runs/               per-run datasets, configs and checkpoints
data/gguf-converted/     GGUF models converted to bf16 checkpoints (reused across runs)
data/tools/              llama.cpp checkout and tools for GGUF export (without Docker)
data/logs/               logs of jobs started from the app
outputs/models/<tag>/    fine-tuned weights + metadata.json + README.md (model card) + gguf/ export
```

## Tests

```bash
pytest
```
