# Translation_Model

Comparing different model translation ability and fine tune the model for special language.

**subtune** fine-tunes text-to-text translation models (TranslateGemma, Qwen3, Seed-X, Hunyuan-MT,
or any Hugging Face chat model) on parallel movie and TV subtitles from OpenSubtitles, through a
Gradio web app.

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

## Setup

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
| `HF_TOKEN` | Hugging Face token. TranslateGemma is gated: accept its terms on the model page first. |
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

| Model | Notes |
|---|---|
| TranslateGemma 4B / 12B / 27B | Gated (Gemma terms). Prompted with its language-code chat template. |
| Qwen3 4B / 8B / 14B | Apache-2.0. Thinking is turned off in the prompt. |
| Seed-X PPO 7B | OpenMDW license. Raw prompt ending in a target-language tag. |
| Hunyuan-MT 7B | Tencent Hunyuan Community License. Uses its documented prompts. |
| Custom | Any Hugging Face causal LM; uses its chat template if it has one. |

Training-memory estimates are shown in the app. On the 32 GB R9700, LoRA fits models up to about
12B, and 27B needs QLoRA or a Modal GPU. QLoRA on ROCm depends on bitsandbytes support for the
card; the app checks it before starting and suggests Modal if it doesn't work.

## Layout

```
data/catalog.sqlite      what OpenSubtitles has, what is downloaded and aligned
data/subtitles/<lang>/   downloaded .srt files, one per OpenSubtitles file id
data/pairs/<src>__<tgt>/ aligned sentence pairs per movie/episode (JSONL)
data/runs/               per-run datasets, configs and checkpoints
data/logs/               logs of jobs started from the app
outputs/models/<tag>/    fine-tuned weights + metadata.json + README.md (model card)
```

## Tests

```bash
pytest
```
