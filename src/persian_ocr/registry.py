"""
Model registry — single source of truth for all 26 OCR models.

Each entry describes one model's repo ID, adapter backend, dependency
profile, and download/inference metadata.  This is the only place where
model-specific constants live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


DownloadKind = Literal["huggingface", "custom", "none"]


@dataclass(frozen=True)
class ModelSpec:
    """Immutable specification for one model in the benchmark."""

    # Identity
    id: str
    display_name: str

    # Adapter — dotted path to a class inheriting OCRAdapter
    adapter: str

    # Dependency profile (pyproject.toml [project.optional-dependencies])
    profile: str

    # How to obtain the model weights
    download_kind: DownloadKind = "huggingface"
    repo_id: str | None = None
    revision: str = "main"
    allow_patterns: tuple[str, ...] = ()
    ignore_patterns: tuple[str, ...] = ()

    # Inference
    prompt: str | None = None
    trust_remote_code: bool = False

    # Extra config passed to the adapter constructor
    extra: dict[str, Any] = field(default_factory=dict)

    # Metadata
    official_persian: bool = False
    persian_evidence: str = ""


# ═══════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════

MODELS: dict[str, ModelSpec] = {}

def _reg(spec: ModelSpec) -> ModelSpec:
    MODELS[spec.id] = spec
    return spec


# ── Traditional OCR engines ──────────────────────────────────────────

_reg(ModelSpec(
    id="ppocrv5_fa",
    display_name="PP-OCRv5 Persian (PaddleOCR)",
    adapter="persian_ocr.adapters.paddle:PaddleOCRAdapter",
    profile="paddle",
    download_kind="custom",
    extra={"lang": "fa"},
    official_persian=True,
    persian_evidence="PP-OCRv5 Arabic/Persian language table explicitly lists fa",
))

_reg(ModelSpec(
    id="easyocr_fa",
    display_name="EasyOCR Persian",
    adapter="persian_ocr.adapters.traditional:EasyOCRAdapter",
    profile="traditional",
    download_kind="custom",
    extra={"languages": ["fa", "en"]},
    official_persian=True,
    persian_evidence="Multilingual model with Persian language support",
))

_reg(ModelSpec(
    id="tesseract_fas",
    display_name="Tesseract Persian (fas)",
    adapter="persian_ocr.adapters.traditional:TesseractAdapter",
    profile="traditional",
    download_kind="custom",
    extra={"language": "fas"},
    official_persian=True,
    persian_evidence="Tesseract Persian language pack (fas)",
))

_reg(ModelSpec(
    id="kraken_fas",
    display_name="Kraken Persian (fas/Arab)",
    adapter="persian_ocr.adapters.traditional:KrakenAdapter",
    profile="traditional",
    download_kind="custom",
    extra={"language": "fas", "script": "Arab"},
    official_persian=True,
    persian_evidence="Kraken Persian-language recognition model",
))

_reg(ModelSpec(
    id="hezarai_crnn_fa_v2",
    display_name="HezarAI CRNN Base FA v2",
    adapter="persian_ocr.adapters.traditional:HezarAdapter",
    profile="traditional",
    download_kind="huggingface",
    repo_id="hezarai/crnn-base-fa-v2",
    official_persian=True,
    persian_evidence="Persian-specific CRNN model from Hezar AI",
))

# ── Document VLMs with official Persian support ─────────────────────

_reg(ModelSpec(
    id="surya-ocr-2",
    display_name="Surya OCR 2",
    adapter="persian_ocr.adapters.surya:SuryaAdapter",
    profile="surya",
    repo_id="datalab-to/surya-ocr-2",
    ignore_patterns=("*.md", "*.png", "*.jpg", "*.git*"),
    official_persian=True,
    persian_evidence="Official 91-language benchmark: fa Persian 82.3%",
))

_reg(ModelSpec(
    id="chandra-ocr-2",
    display_name="Chandra OCR 2",
    adapter="persian_ocr.adapters.chandra:ChandraAdapter",
    profile="surya",
    repo_id="datalab-to/chandra-ocr-2",
    ignore_patterns=("*.md", "*.png", "*.jpg", "*.git*"),
    official_persian=True,
    persian_evidence="Official 90-language benchmark: fa 75.1%",
))

# ── Persian-specific fine-tunes ──────────────────────────────────────

_reg(ModelSpec(
    id="qwen3_vl_persian_arabic_ocr",
    display_name="Qwen3-VL-2B Persian/Arabic OCR",
    adapter="persian_ocr.adapters.transformers_vlm:QwenVLAdapter",
    profile="transformers",
    repo_id="mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0",
    prompt="Extract all visible text from this image. Return only the transcription.",
    official_persian=True,
    persian_evidence="Persian-specific fine-tune of Qwen3-VL-2B",
))

_reg(ModelSpec(
    id="weightedai_persian_ocr",
    display_name="WeightedAI Persian OCR",
    adapter="persian_ocr.adapters.transformers_vlm:WeightedAIAdapter",
    profile="transformers",
    repo_id="WeightedAI/Persian_OCR",
    prompt="Extract all visible text from this image. Return only the transcription.",
    official_persian=True,
    persian_evidence="Persian-specific OCR model",
))

_reg(ModelSpec(
    id="khanandeh",
    display_name="Khanandeh 0.1 Persian OCR 2B Instruct",
    adapter="persian_ocr.adapters.transformers_vlm:KhanandehAdapter",
    profile="transformers",
    download_kind="custom",  # needs adapter + base model
    repo_id="oddadmix/Khanandeh-0.1-Persian-OCR-2B-Instruct",
    extra={
        "base_repo_id": "unsloth/qwen2-vl-2b-instruct-unsloth-bnb-4bit",
    },
    prompt="Extract all visible text from this image. Return only the transcription.",
    official_persian=True,
    persian_evidence="Persian OCR 2B Instruct model",
))

# ── Generic document VLMs ────────────────────────────────────────────

_reg(ModelSpec(
    id="deepseek-ocr",
    display_name="DeepSeek OCR",
    adapter="persian_ocr.adapters.deepseek:DeepSeekOCRAdapter",
    profile="deepseek",
    repo_id="deepseek-ai/DeepSeek-OCR",
    prompt="<image>\n<|grounding|>Convert the document to markdown.",
    trust_remote_code=True,
    extra={"base_size": 768, "image_size": 1344, "crop_mode": True, "test_compress": True},
))

_reg(ModelSpec(
    id="deepseek-ocr-2",
    display_name="DeepSeek OCR 2",
    adapter="persian_ocr.adapters.deepseek:DeepSeekOCR2Adapter",
    profile="deepseek",
    repo_id="deepseek-ai/DeepSeek-OCR-2",
    prompt="<image>\n<|grounding|>Convert the document to markdown.",
    trust_remote_code=True,
    extra={"base_size": 768, "image_size": 1344, "crop_mode": True, "test_compress": True},
))

_reg(ModelSpec(
    id="dots_mocr",
    display_name="Dots MOCR",
    adapter="persian_ocr.adapters.transformers_vlm:GenericVLMAdapter",
    profile="transformers",
    repo_id="rednote-hilab/dots.mocr",
    prompt="<image>\nExtract all visible text from this image.",
))

_reg(ModelSpec(
    id="dots_ocr",
    display_name="Dots OCR",
    adapter="persian_ocr.adapters.transformers_vlm:GenericVLMAdapter",
    profile="transformers",
    repo_id="rednote-hilab/dots.ocr",
    prompt="<image>\nExtract all visible text from this image.",
))

_reg(ModelSpec(
    id="falcon_ocr",
    display_name="Falcon OCR",
    adapter="persian_ocr.adapters.transformers_vlm:GenericVLMAdapter",
    profile="transformers",
    repo_id="tiiuae/Falcon-OCR",
    prompt="<image>\nExtract all visible text from this image.",
))

_reg(ModelSpec(
    id="glm_ocr",
    display_name="GLM OCR",
    adapter="persian_ocr.adapters.transformers_vlm:GenericVLMAdapter",
    profile="transformers",
    repo_id="zai-org/GLM-OCR",
    prompt="<image>\nExtract all visible text from this image.",
))

_reg(ModelSpec(
    id="got_ocr2",
    display_name="GOT OCR 2.0 HF",
    adapter="persian_ocr.adapters.transformers_vlm:GenericVLMAdapter",
    profile="transformers",
    repo_id="stepfun-ai/GOT-OCR-2.0-hf",
    prompt="<image>\nExtract all visible text from this image.",
    trust_remote_code=True,
))

_reg(ModelSpec(
    id="hunyuan_ocr",
    display_name="Hunyuan OCR",
    adapter="persian_ocr.adapters.transformers_vlm:GenericVLMAdapter",
    profile="transformers",
    repo_id="tencent/HunyuanOCR",
    prompt="<image>\nExtract all visible text from this image.",
))

_reg(ModelSpec(
    id="infinity_parser2_flash",
    display_name="Infinity Parser 2 Flash",
    adapter="persian_ocr.adapters.transformers_vlm:GenericVLMAdapter",
    profile="transformers",
    repo_id="infly/Infinity-Parser2-Flash",
    prompt="<image>\nExtract all visible text from this image.",
))

_reg(ModelSpec(
    id="infinity_parser2_pro",
    display_name="Infinity Parser 2 Pro",
    adapter="persian_ocr.adapters.transformers_vlm:GenericVLMAdapter",
    profile="transformers",
    repo_id="infly/Infinity-Parser2-Pro",
    prompt="<image>\nExtract all visible text from this image.",
))

_reg(ModelSpec(
    id="kdl_frontier_parser_nano",
    display_name="KDL Frontier Parser Nano",
    adapter="persian_ocr.adapters.transformers_vlm:GenericVLMAdapter",
    profile="transformers",
    repo_id="KDLAI/KDL-Frontier-Parser-nano",
    prompt="<image>\nExtract all visible text from this image.",
))

_reg(ModelSpec(
    id="lightonocr2_1b",
    display_name="LightOn OCR 2 1B",
    adapter="persian_ocr.adapters.transformers_vlm:GenericVLMAdapter",
    profile="transformers",
    repo_id="lightonai/LightOnOCR-2-1B",
    prompt="<image>\nExtract all visible text from this image.",
))

_reg(ModelSpec(
    id="mineru25_pro",
    display_name="MinerU 2.5 Pro",
    adapter="persian_ocr.adapters.transformers_vlm:GenericVLMAdapter",
    profile="transformers",
    repo_id="opendatalab/MinerU2.5-Pro-2605-1.2B",
    prompt="<image>\nExtract all visible text from this image.",
))

_reg(ModelSpec(
    id="nanonets_ocr2_15b",
    display_name="Nanonets OCR2 1.5B exp",
    adapter="persian_ocr.adapters.transformers_vlm:GenericVLMAdapter",
    profile="transformers",
    repo_id="nanonets/Nanonets-OCR2-1.5B-exp",
    prompt="<image>\nExtract all visible text from this image.",
))

_reg(ModelSpec(
    id="paddleocr_vl",
    display_name="PaddleOCR VL 0.9B",
    adapter="persian_ocr.adapters.transformers_vlm:GenericVLMAdapter",
    profile="transformers",
    repo_id="lvyufeng/PaddleOCR-VL-0.9B",
    prompt="<image>\nExtract all visible text from this image.",
))

_reg(ModelSpec(
    id="unlimited_ocr_gguf",
    display_name="Unlimited OCR GGUF Q4_K_M",
    adapter="persian_ocr.adapters.gguf:UnlimitedOCRAdapter",
    profile="gguf",
    repo_id="sahilchachra/Unlimited-OCR-GGUF",
    allow_patterns=("Unlimited-OCR-Q4_K_M.gguf", "mmproj-Unlimited-OCR-F16.gguf"),
    extra={"quant": "Q4_K_M"},
))


# ═══════════════════════════════════════════════════════════════════════
# Priority order (Persian support descending)
# ═══════════════════════════════════════════════════════════════════════

PRIORITY = [
    "ppocrv5_fa",          # 1  — official Persian
    "surya-ocr-2",         # 2  — official Persian 82.3%
    "chandra-ocr-2",       # 3  — official Persian 75.1%
    "qwen3_vl_persian_arabic_ocr",  # 4
    "weightedai_persian_ocr",       # 5
    "khanandeh",           # 6
    "hezarai_crnn_fa_v2",  # 7  — Persian-specific CRNN
    "easyocr_fa",          # 8
    "tesseract_fas",       # 9
    "kraken_fas",          # 10
    "deepseek-ocr-2",      # 11
    "deepseek-ocr",        # 12
    "got_ocr2",            # 13
    "glm_ocr",             # 14
    "dots_ocr",            # 15
    "dots_mocr",           # 16
    "hunyuan_ocr",         # 17
    "falcon_ocr",          # 18
    "mineru25_pro",        # 19
    "kdl_frontier_parser_nano",  # 20
    "infinity_parser2_flash",    # 21
    "infinity_parser2_pro",      # 22
    "nanonets_ocr2_15b",   # 23
    "lightonocr2_1b",      # 24
    "paddleocr_vl",        # 25
    "unlimited_ocr_gguf",  # 26
]
