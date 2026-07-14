# Comprehensive Report — Community & Social-Media Discussion of OCR Model Training (Fine-tuning / Training / Post-training / Pre-training)

**Prepared:** 2026-07-14 · **Scope:** highest-signal practical advice, all scripts/languages · **Trigger:** user request in `persian-ocr` project dir
**Method:** local SearXNG (`localhost:8888`, restarted this session) + `web_extract` + targeted Jina-Reader proxy for walled Medium. Audit files (sources.md, claim_cards.md) sit alongside this report.

---

## 1. TL;DR (the decision-relevant conclusion)

The practitioner consensus across Hugging Face, NVIDIA, Hacker News, GitHub, and tutorials converges on six points:

1. **Data > architecture.** The limiting factor for a good OCR model is annotated data, not the model. Synthetic data at scale is now the default way to get there, and the recipe is **language-agnostic** (NVIDIA Nemotron OCR v2 used a SynthDoG-style renderer + the mOSCAR multilingual text corpus to produce ~12M synthetic multilingual samples).
2. **Default choice in 2025-2026 is an OCR-fine-tuned VLM** (Qwen2.5-VL/3-VL, OlmOCR, DeepSeek-OCR, Nanonets-OCR, PaddleOCR-VL, GLM-OCR, Dots.OCR, Nemotron OCR v2). "OCR-free" models (Donut) skip a recognition engine entirely.
3. **Evaluate the base model before fine-tuning.** Only fine-tune when it underperforms, and expect gains proportional to how far your domain sits from the pretraining distribution.
4. **Prefer LoRA/PEFT over full fine-tuning for small/low-resource adaptation** — full FT on a few hundred pages easily overwrites pretrained knowledge (catastrophic forgetting), per a widely-cited r/LocalLLaMA Marathi Qwen2.5-VL thread.
5. **VLM-based OCR hallucinates.** "Smart priors" make wrong content plausible; dangerous for finance/legal/verbatim text. Keep a deterministic recognition stage where fidelity matters, and treat any LLM refinement as risky.
6. **For Persian/low-resource specifically:** (a) pipeline auto-translation is a real bug — suppress it with explicit "never translate / keep source language" prompts; (b) non-Latin tokenizer inefficiency can cap SFT gains — consider continued pretraining or a better tokenizer, not just SFT; (c) go evaluation-first: build a private eval set for the exact missing capability before generating data.

**Directly useful to your `persian-ocr` work:** DeepSeek-OCR is repeatedly singled out for strong Persian/Arabic support + 3B consumer-runnable + dynamic cropping; a published Unsloth LoRA recipe fine-tuned it on the *Parsynth OCR* (200k synthetic Persian) dataset. NVIDIA's open multilingual synthetic pipeline is the template to reproduce for Persian if you need training data at scale.

---

## 2. What I could and could not access (transparency on evidence)

| Platform | Status | How |
|----------|--------|-----|
| Hugging Face blog + forums | ✅ Full | `web_extract` |
| NVIDIA / org blogs | ✅ Full | `web_extract` |
| Hacker News | ✅ Full (1 thread), ⚠️ 1 rate-limited | `web_extract` |
| GitHub Discussions/Issues (PaddleOCR, TrOCR) | ✅ Full (frame-rendered) | `web_extract` |
| Practitioner tutorials (freeCodeCamp, HackerNoon, Towards AI, Medium) | ✅ Full (Medium via Jina proxy) | `web_extract` / Jina |
| **Reddit** | ⚠️ **Snippet only** | Bot-walled: browser + Jina proxy + curl all returned Reddit's "blocked by network security" 403. `reddit_rss` MCP not loaded this session. Captured via SearXNG `site:reddit.com` result snippets. |
| **X / Twitter** | ❌ **Inaccessible** | Returns only profile pages; posts behind login wall. No usable content. |
| **LinkedIn** | ⚠️ **Snippet only** | Accessible as search-result descriptions; full post bodies not retrievable. |

**Consequence:** Reddit/X/LinkedIn claims below are flagged "snippet" in the claim cards and should be treated as leads to verify, not as verified statements. The substantive analysis rests on HF/NVIDIA/HN/GitHub/tutorials, which are substantial and mutually reinforcing.

*(Environment note: your local SearXNG was down at session start; I restarted it with `docker compose up -d` in `C:\Services\searxng` — it now serves on `127.0.0.1:8888`.)*

---

## 3. The landscape: what "training an OCR model" means now

Community usage splits into four layers (not mutually exclusive):

- **Pre-training / from-scratch:** rare for individuals; NVIDIA's Nemotron OCR v2 is the prominent open example — built on a *generic synthetic data pipeline* (text from mOSCAR → rendered with a modified SynthDoG → millions of image-text pairs). Takeaway: you don't need hand labels at the millions scale; you need a good renderer + a text corpus in your language.
- **Continued pre-training / domain adaptation:** relevant for low-resource languages where the base tokenizer/model under-represents the script (HF forum, C11). More than SFT when the script itself is under-trained.
- **Fine-tuning (full vs PEFT):** the everyday task. PaddleOCR/EasyOCR/TrOCR for detector+recognizer stacks; Qwen-VL/OlmOCR/DeepSeek-OCR for VLM routes. LoRA is the community default for small adapters.
- **Post-training / inference-time prompting:** VLM OCR is largely *prompted* ("Free OCR."), and "post-training" for OCR-VLMs means either LoRA or RL-based reasoning (e.g., DianJin-OCR-R1, DocVLM). The boundary between "pretraining data" and "post-training data" is explicitly blurring (LinkedIn trend snippet, S18-related).

---

## 4. Highest-signal practical advice (synthesized)

### 4.1 Decide *whether* to fine-tune
- **Evaluate the base model on your real data first** (Reddit r/computervision, C6). Fine-tuning pays off when your domain is far from the pretraining distribution (a niche font, a specific script, noisy scans) and is wasted otherwise.
- **Match the method to data volume:**
  - Hundreds of pages / a niche script → **LoRA/PEFT**, not full FT (C4). Full FT on ~700 pages overwrote a Qwen2.5-VL's knowledge in a reported case.
  - Thousands+ clean pairs → full FT or continued pretraining may be justified.
  - Zero labels → **synthetic generation** (C14) + a small real eval set.

### 4.2 Data: synthetic-first, eval-driven
- The dominant 2025-2026 pattern is **synthetic data at scale** (Nemotron, OlmOCR, Parsynth, EasyOCR tutorials). Recipe: take a real text corpus → render with a document/font/background renderer (SynthDoG-style) → augment (brightness, noise, perspective, blur, contrast, saturation).
- **Language-agnostic:** the same pipeline works for Persian if you have a Persian text corpus (mOSCAR has Persian; OpenITI MAKHZAN covers Arabic/Persian/Ottoman/Urdu ground truth). This is the single most actionable insight for `persian-ocr`.
- **Evaluation-first for low-resource (C10):** define the exact missing capability + a private eval set *before* generating data. "Don't build more Persian data blindly."
- **Watch the domain gap:** synthetic data drifts from real scans; always hold out real annotated pages for final eval.

### 4.3 Training mechanics (framework-specific, C13)
- **PaddleOCR:** detection expects 8-point bbox coords + cropped recognition regions; choose PP-OCRv4/v3 pretrained; separate detection vs recognition configs.
- **EasyOCR:** convert dataset to **LMDB**; built on the ClovaAI deep-text-recognition-benchmark; fine-tune the recognition model.
- **TrOCR:** needs `(image, text)` pairs; follow the repo notebook.
- **VLM (Qwen-VL / DeepSeek-OCR / OlmOCR):** prepare conversational samples (image + "Free OCR." → transcription); LoRA `r=16`, target q/k/v/o + MLP; train on consumer GPU (T4) in minutes for small adapters (C5).

### 4.4 Risks the community keeps raising
- **Hallucination (C7):** VLM "smart priors" can silently correct wrong. For verbatim/legal/financial text, keep a deterministic recognition stage; treat LLM refinement as lossy.
- **Prompt injection (C8):** OCR over untrusted docs can embed instructions that poison downstream training data. Mitigate with strict JSON I/O separation + sandboxing.
- **Auto-translation (C9):** multilingual pipelines translate non-English text by default. Explicitly forbid translation and keep the source script — critical for Persian.
- **Forgetting (C4):** full FT on small data destroys prior knowledge; LoRA + a frozen base is safer.

### 4.5 Model selection (open-weight bias, C3/C15)
Community prefers open-weight models for **cost + privacy** in self-hosted pipelines (HF blog). Cloud APIs (Google Vision, MathPix, Gemini) are still used *inside* pipelines as a quality boost, then refined. For Persian specifically, **DeepSeek-OCR** is the most-cited open model with strong Persian/Arabic support.

---

## 5. Conflicting / uncertain views

- **Full FT vs PEFT:** r/LocalLLaMA shows both a catastrophic-forgetting warning (Marathi, full FT) and a success story (Qwen3.5-0.8B full FT "mainly teaches structure/format, not overwrite"). Resolution: full FT can work when the goal is *output formatting*, not *new script knowledge*; for new-script learning on small data, PEFT is safer.
- **"OCR is solved / use a VLM" vs "OCR is risky / use deterministic":** HN debate (C7) pushes back hard on LLM-OCR for fidelity-critical use. Both are right in different regimes — VLM for understanding/layout, deterministic for verbatim.
- **Benchmark representativeness (C16):** OmniDocBench / olmOCR-bench skew English/Chinese; a model "SOTA" there may still fail on Persian. Low-resource coverage is thin — build your own eval.
- **Vendor multilingual claims (C12):** "DeepSeek-OCR on par with English for Persian/Arabic" is author/vendor-stated, not independently benchmarked in the sources; verify on your data.

---

## 6. Directed recommendations for `persian-ocr`

1. **Start from a Persian-capable open VLM base** (DeepSeek-OCR 3B, or Qwen2.5-VL/3-VL) rather than training from scratch.
2. **Build a synthetic Persian pipeline** mirroring Nemotron: Persian text corpus (mOSCAR-Persian / OpenITI / Matina Persian corpus) → SynthDoG-style renderer → augment. Seed with the existing *Parsynth OCR 200k* dataset.
3. **LoRA, not full FT**; hold out real Persian pages for eval; track CER/normalised edit distance, not just loss.
4. **Hard-code "do not translate; preserve Persian script"** in every prompt; sandbox any LLM refine stage.
5. **If accuracy stalls**, check tokenizer compression (Persian tokens per char) before blaming data — continued pretraining or a better tokenizer may be the fix (C11).

---

## 7. Citations (by source)

- **S1** Hugging Face — "Supercharge your OCR Pipelines with Open Models" (2025-10-21) · https://huggingface.co/blog/ocr-open-models
- **S2** NVIDIA / Ryan Chesler et al. — "Building a Fast Multilingual OCR Model with Synthetic Data" (Nemotron OCR v2) (2026-04-17) · https://huggingface.co/blog/nvidia/nemotron-ocr-v2
- **S3** Hugging Face Forums — "How can i build a High Quality dataset?" (Persian SLM) (2026-06) · https://discuss.huggingface.co/t/how-can-i-build-a-high-quality-dataset/176571
- **S4** Hacker News — "Show HN: OCR pipeline for ML training" (2025-04-05) · https://news.ycombinator.com/item?id=43590998
- **S5** Hacker News — "DeepSeek OCR" (2025) · https://news.ycombinator.com/item?id=45640594 *(rate-limited; snippet only)*
- **S6** Reddit r/LocalLLaMA — "Fine-tuning qwen2.5 vl for Marathi OCR" (2025-07-25) · https://www.reddit.com/r/LocalLLaMA/comments/1m8qtpd/ *(bot-walled; snippet)*
- **S7** Reddit r/LocalLLM — "You can now Fine-tune DeepSeek-OCR locally!" (2025-11-04) · https://www.reddit.com/r/LocalLLM/comments/1ooan40/ *(snippet)*
- **S8** Reddit r/computervision — "When do you recommend finetuning OCR models?" (2026-04-08) · https://www.reddit.com/r/computervision/comments/1sfjmmp/ *(snippet)*
- **S9** Reddit r/LocalLLaMA — "Need help in fine-tuning of OCR model at production level" (2026-03-12) · https://www.reddit.com/r/LocalLLaMA/comments/1rr0evv/ *(snippet)*
- **S10** Reddit r/LocalLLaMA — "Update: I fine-tuned Qwen3.5-0.8B for OCR" (2026-04-14) · https://www.reddit.com/r/LocalLLaMA/comments/1skyq19/ *(snippet)*
- **S11** GitHub PaddlePaddle/PaddleOCR Discussions #14387 (2024-12-15) · https://github.com/PaddlePaddle/PaddleOCR/discussions/14387
- **S12** GitHub microsoft/unilm Issues #627 — Finetune TrOCR (2022-02-18) · https://github.com/microsoft/unilm/issues/627
- **S13** freeCodeCamp — "How to Fine-Tune EasyOCR with a Synthetic Dataset" (2024-01-05) · https://www.freecodecamp.org/news/how-to-fine-tune-easyocr-with-a-synthetic-dataset/
- **S14** HackerNoon — "OCR Fine-Tuning: From Raw Data to Custom Paddle OCR Model" (2025-01-29) · https://hackernoon.com/ocr-fine-tuning-from-raw-data-to-custom-paddle-ocr-model
- **S15** Towards AI — "How To Make a Synthesized Dataset To Fine-Tune Your OCR" (2024-01-25) · https://towardsai.net/p/data-science/how-to-make-a-synthesized-dataset-to-fine-tune-your-ocr
- **S16** Medium (Jina proxy) — "Fine-Tune DeepSeek OCR with Unsloth" (Persian) (2025-11-05) · https://medium.com/@matteo28/how-to-fine-tune-the-new-deepseek-ocr-model-with-unsloth-3964c01f39bd
- **S17** HF community — "Hall of Multimodal OCR VLMs" (2025-10-31) · https://huggingface.co/blog/prithivMLmods/multimodal-ocr-vlms
- **S18** LinkedIn (snippet) — Nemotron OCR v2, ~12M synthetic multilingual (2026-04-17) · https://www.linkedin.com/posts/ryan-chesler_...7450951271525646338
- **S19** X/Twitter — inaccessible (bot-wall).
- **S20** LinkedIn (snippet) — EasyOCR synthetic fine-tune (freeCodeCamp repost) (2024-01) · https://www.linkedin.com/posts/free-code-camp_...7149445193417539584

---

## 8. Next actions / gaps

- **To upgrade Reddit/X/LinkedIn evidence:** load the `reddit_rss` MCP (no-auth, returns real posts) for full Reddit threads, and retry X/LinkedIn via authenticated sessions or a logged-in browser profile. I can do this on request.
- **Verification needed:** the DeepSeek-OCR Persian "+88.64% understanding" (S7) and the CER 23%→6% (S16) are single-source/anecdotal — validate on a real Persian eval set before relying on them.
- **Build the eval set first** (C10) for `persian-ocr`; that unlocks every downstream training decision with evidence instead of guesswork.
