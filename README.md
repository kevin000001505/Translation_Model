# Translation_Model

Comparing different model translation ability and fine tune the model for special language.

**subtune** fine-tunes text-to-text translation models (Hy-MT2, Hunyuan-MT, Seed-X, Qwen3, or any
public Hugging Face chat model) on parallel movie and TV subtitles from OpenSubtitles, through a
Gradio web app. Every model it offers downloads without a token or an access request.

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
| Hy-MT2 7B | Apache-2.0. Tencent's translation model; uses its documented prompts. |
| Hunyuan-MT 7B | Tencent Hunyuan Community License (check its territory restrictions). |
| Seed-X PPO 7B | OpenMDW license. Raw prompt ending in a target-language tag. |
| Qwen3 4B / 8B / 14B | Apache-2.0. Thinking is turned off in the prompt. |
| Custom | Any public causal LM. The app checks Hugging Face and refuses gated or private models. |

Gated models such as Google's TranslateGemma and Gemma 3 are left out because they require an
access request.

Training-memory estimates are shown in the app. On the 32 GB R9700, LoRA fits the 4B–8B models;
Qwen3 14B needs QLoRA or a Modal GPU. QLoRA on ROCm depends on bitsandbytes support for the
card; the app checks it before starting and suggests Modal if it doesn't work.

## Cloud costs (Modal)

The GPU is billed per second only while a job runs. A job is shut down:

- when training finishes or fails: the job runs as an ephemeral Modal app, which stops as soon as
  the launcher exits, and the GPU container is released 2 seconds after it returns (Modal's
  default keeps it idle for up to 60 seconds);
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
data/logs/               logs of jobs started from the app
outputs/models/<tag>/    fine-tuned weights + metadata.json + README.md (model card)
```

## Tests

```bash
pytest
```
