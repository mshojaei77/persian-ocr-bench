# Source Ledger — OCR Model Training Community/SM Sweep

Scope: fine-tuning / training / post-training / pre-training of OCR models.
Recency: emphasis on 2024-2026, evergreen classics retained.
Reachability: see report §"What I could and could not access".

| ID | Source | Type | Date | URL | Status |
|----|--------|------|------|-----|--------|
| S1 | Hugging Face blog — "Supercharge your OCR Pipelines with Open Models" | Practitioner/Org blog | 2025-10-21 | https://huggingface.co/blog/ocr-open-models | full |
| S2 | NVIDIA (Ryan Chesler et al.) HF blog — Nemotron OCR v2, synthetic multilingual | Practitioner/Org blog | 2026-04-17 | https://huggingface.co/blog/nvidia/nemotron-ocr-v2 | full |
| S3 | HF Forums — "How can i build a High Quality dataset?" (Persian SLM) | Forum thread | 2026-06-05/07 | https://discuss.huggingface.co/t/how-can-i-build-a-high-quality-dataset/176571 | full |
| S4 | Hacker News — "Show HN: OCR pipeline for ML training (tables, diagrams, math, multilingual)" | Forum/comments | 2025-04-05 | https://news.ycombinator.com/item?id=43590998 | full |
| S5 | Hacker News — "DeepSeek OCR" | Forum thread | (2025) | https://news.ycombinator.com/item?id=45640594 | snippet only (429 rate limit) |
| S6 | Reddit r/LocalLLaMA — "Fine-tuning qwen2.5 vl for Marathi OCR" | Forum thread | 2025-07-25 | https://www.reddit.com/r/LocalLLaMA/comments/1m8qtpd/ | snippet only (bot-walled) |
| S7 | Reddit r/LocalLLM — "You can now Fine-tune DeepSeek-OCR locally!" | Forum thread | 2025-11-04 | https://www.reddit.com/r/LocalLLM/comments/1ooan40/ | snippet only (bot-walled) |
| S8 | Reddit r/computervision — "When do you recommend finetuning OCR models?" | Forum thread | 2026-04-08 | https://www.reddit.com/r/computervision/comments/1sfjmmp/ | snippet only (bot-walled) |
| S9 | Reddit r/LocalLLaMA — "Need help in fine-tuning of OCR model at production level" | Forum thread | 2026-03-12 | https://www.reddit.com/r/LocalLLaMA/comments/1rr0evv/ | snippet only (bot-walled) |
| S10 | Reddit r/LocalLLaMA — "Update: I fine-tuned Qwen3.5-0.8B for OCR" | Forum thread | 2026-04-14 | https://www.reddit.com/r/LocalLLaMA/comments/1skyq19/ | snippet only (bot-walled) |
| S11 | GitHub PaddlePaddle/PaddleOCR Discussions #14387 — finetuning questions | GitHub Discussion | 2024-12-15 | https://github.com/PaddlePaddle/PaddleOCR/discussions/14387 | full (partial render) |
| S12 | GitHub microsoft/unilm Issues #627 — Finetune TrOCR on own dataset | GitHub Issue | 2022-02-18 | https://github.com/microsoft/unilm/issues/627 | full (partial render) |
| S13 | freeCodeCamp — "How to Fine-Tune EasyOCR with a Synthetic Dataset" | Tutorial | 2024-01-05 | https://www.freecodecamp.org/news/how-to-fine-tune-easyocr-with-a-synthetic-dataset/ | full |
| S14 | HackerNoon — "OCR Fine-Tuning: From Raw Data to Custom Paddle OCR Model" | Tutorial | 2025-01-29 | https://hackernoon.com/ocr-fine-tuning-from-raw-data-to-custom-paddle-ocr-model | full |
| S15 | Towards AI — "How To Make a Synthesized Dataset To Fine-Tune Your OCR" | Tutorial | 2024-01-25 | https://towardsai.net/p/data-science/how-to-make-a-synthesized-dataset-to-fine-tune-your-ocr | full |
| S16 | Medium (via Jina proxy) — "Fine-Tune DeepSeek OCR with Unsloth" (Persian) | Tutorial | 2025-11-05 | https://medium.com/@matteo28/how-to-fine-tune-the-new-deepseek-ocr-model-with-unsloth-3964c01f39bd | full (proxy) |
| S17 | HF community blog — "Hall of Multimodal OCR VLMs" | Practitioner/Org blog | 2025-10-31 | https://huggingface.co/blog/prithivMLmods/multimodal-ocr-vlms | full |
| S18 | LinkedIn (snippet) — Nemotron OCR v2, ~12M synthetic multilingual | Social (snippet) | 2026-04-17 | https://www.linkedin.com/posts/ryan-chesler_...7450951271525646338 | snippet only |
| S19 | X/Twitter — OCR fine-tuning queries | Social | n/a | https://x.com/... | inaccessible (bot-wall); only profile snippets returned |
| S20 | LinkedIn (snippet) — EasyOCR synthetic fine-tune (freeCodeCamp repost) | Social (snippet) | 2024-01 | https://www.linkedin.com/posts/free-code-camp_...7149445193417539584 | snippet only |

## Notes on reachability
- Reddit: browser + Jina proxy + curl all hit Reddit's "blocked by network security" 403. `reddit_rss` MCP not loaded as a callable tool this session. => Reddit captured only via SearXNG `site:reddit.com` result snippets.
- X/Twitter: no usable content; returns only profile pages, full posts bot-walled.
- LinkedIn: accessible only as search-result snippets (no full post bodies).
- SearXNG local instance was down at start; started it via `docker compose up -d` in C:\Services\searxng (now serving on 127.0.0.1:8888).
