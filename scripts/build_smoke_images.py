"""Build the 20-image Persian OCR smoke corpus and its JSONL manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "smoke"
MANIFEST_PATH = ROOT / "data" / "manifest.jsonl"
SOURCE_SCAN = ROOT / "assets" / "sources" / "babak_by_nafisi_page10.jpg"
FONT_DIR = ROOT / "assets" / "fonts"

FONTS = {
    "serif": FONT_DIR / "NotoNaskhArabic.ttf",
    "sans": FONT_DIR / "NotoSansArabic.ttf",
}


@dataclass(frozen=True)
class Sample:
    sample_id: str
    description: str
    reference: str
    font: str = "serif"
    size: int = 48
    attributes: tuple[str, ...] = ()
    primary_stratum: str = "clean_modern_printed"
    origin: str = "synthetic"
    method: str = "rendered"
    numeric_spans: tuple[tuple[str, str, str], ...] = field(default_factory=tuple)


SAMPLES = (
    Sample("smoke_001", "Simple clean Persian sentence", "امروز هوا روشن و دلپذیر است."),
    Sample("smoke_002", "Clean serif Persian font", "کتاب خوب پنجره‌ای رو به دانایی است.", "serif"),
    Sample("smoke_003", "Clean sans-serif font", "زندگی با امید و تلاش زیباتر می‌شود.", "sans"),
    Sample("smoke_004", "Small font", "این جمله با اندازهٔ کوچک چاپ شده است.", "sans", 25, ("small_font",)),
    Sample("smoke_005", "Bold text with punctuation", "هشدار: در را ببندید؛ متشکرم!", "sans", 52, ("bold", "punctuation")),
    Sample("smoke_006", "Real scanned book line", "روشن بینی و تیزبینی و دوراندیشی کامل و ابرام و پشت کار شگرف", attributes=("real_scan",), primary_stratum="real_scanned_documents", origin="real", method="public_domain_scan_crop"),
    Sample("smoke_007", "Low-resolution JPEG line", "کیفیت پایین تصویر نباید متن را پنهان کند.", "sans", 42, ("jpeg", "low_resolution"), method="jpeg_roundtrip"),
    Sample("smoke_008", "Slightly skewed phone photograph", "این تصویر با تلفن همراه گرفته شده است.", "sans", 43, ("mobile_photo", "skew"), primary_stratum="mobile_camera_photographs", method="phone_photo_simulation"),
    Sample("smoke_009", "Neat Persian handwriting", "هر روز چند سطر با خط خوانا می‌نویسم.", "serif", 51, ("handwriting",), primary_stratum="persian_handwriting", method="synthetic_handwriting_simulation"),
    Sample("smoke_010", "Cursive handwriting", "دوستی قدیمی برایم نامه‌ای صمیمی نوشت.", "serif", 55, ("handwriting", "cursive"), primary_stratum="persian_handwriting", method="synthetic_cursive_simulation"),
    Sample("smoke_011", "ZWNJ-heavy text", "می‌روم، خانه‌ها و متنِ گفته‌شده را می‌بینم.", "serif", 47, ("zwnj", "punctuation")),
    Sample("smoke_012", "Joining and spacing challenge", "به آن‌ها گفتم کتاب‌ها را دسته‌بندی کنند.", "serif", 47, ("zwnj", "spacing", "joining")),
    Sample("smoke_013", "Persian digits", "شماره‌ها: ۱۲۳۴۵۶۷۸۹۰", "sans", 52, ("persian_digits",), primary_stratum="numeric_heavy", numeric_spans=(("۱۲۳۴۵۶۷۸۹۰", "integer", "1234567890"),)),
    Sample("smoke_014", "Latin digits inside Persian text", "کد رهگیری سفارش 2026 برابر 48371 است.", "sans", 47, ("latin_digits", "mixed_script"), primary_stratum="numeric_heavy", numeric_spans=(("2026", "integer", "2026"), ("48371", "integer", "48371"))),
    Sample("smoke_015", "Date, time, price, percentage", "در ۱۴۰۵/۰۴/۲۳ ساعت ۰۹:۳۰، قیمت ۲۵۰٬۰۰۰ تومان با ۱۵٪ تخفیف بود.", "sans", 43, ("persian_digits", "date", "time", "price", "percentage", "punctuation"), primary_stratum="numeric_heavy", numeric_spans=(("۱۴۰۵/۰۴/۲۳", "date", "1405/04/23"), ("۰۹:۳۰", "time", "09:30"), ("۲۵۰٬۰۰۰", "price", "250000"), ("۱۵٪", "percentage", "15%"))),
    Sample("smoke_016", "Mixed Persian and English", "نسخهٔ جدید Persian OCR آماده است.", "sans", 48, ("mixed_script", "english"), primary_stratum="mixed_persian_english"),
    Sample("smoke_017", "Email, URL, abbreviation, parentheses", "نشانی (AI): test@example.com و example.org/fa است.", "sans", 41, ("mixed_script", "email", "url", "parentheses", "abbreviation"), primary_stratum="mixed_persian_english"),
    Sample("smoke_018", "Long dense Persian line", "پژوهشگران برای سنجش دقیق سامانه‌های بازشناسی متن، نمونه‌های گوناگون و دشوار را با دقت بررسی می‌کنند.", "serif", 39, ("long_line", "dense")),
    Sample("smoke_019", "Historical font or bleed-through", "همواره یکی از خصایص ملت ما بوده است", attributes=("historical", "bleed_through", "real_scan"), primary_stratum="historical_or_naturally_degraded", origin="real", method="public_domain_scan_crop_with_bleed_simulation"),
    Sample("smoke_020", "Blank or non-text image", "", "sans", 48, ("blank", "non_text"), primary_stratum="blank_non_text", method="deterministic_non_text"),
)


def visual_text(text: str) -> str:
    return get_display(arabic_reshaper.reshape(text), base_dir="R")


def render_line(sample: Sample) -> Image.Image:
    font = ImageFont.truetype(FONTS[sample.font], sample.size)
    shown = visual_text(sample.reference)
    probe = Image.new("RGB", (16, 16), "white")
    bbox = ImageDraw.Draw(probe).textbbox((0, 0), shown, font=font, stroke_width=1 if sample.sample_id == "smoke_005" else 0)
    width = max(360, bbox[2] - bbox[0] + 64)
    height = max(88, bbox[3] - bbox[1] + 42)
    image = Image.new("RGB", (width, height), (250, 249, 246))
    draw = ImageDraw.Draw(image)
    draw.text((width - 32, height // 2), shown, font=font, fill=(22, 22, 22), anchor="rm", stroke_width=1 if sample.sample_id == "smoke_005" else 0)
    return image


def scanned_crop(sample_id: str) -> Image.Image:
    page = Image.open(SOURCE_SCAN).convert("L")
    # Page 10, Saeed Nafisi's public-domain 1953 book "Babak".
    boxes = {
        "smoke_006": (105, 421, 875, 460),
        "smoke_019": (455, 465, 885, 501),
    }
    crop = page.crop(boxes[sample_id])
    crop = ImageEnhance.Contrast(crop).enhance(1.08)
    if sample_id == "smoke_019":
        ghost = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT).filter(ImageFilter.GaussianBlur(1.2))
        crop = Image.blend(crop, ghost, 0.08)
    return crop.convert("RGB")


def transform(sample: Sample, image: Image.Image) -> Image.Image:
    if sample.sample_id in {"smoke_009", "smoke_010"}:
        shear = -0.08 if sample.sample_id == "smoke_009" else -0.15
        extra = int(abs(shear) * image.height) + 8
        image = image.transform(
            (image.width + extra, image.height),
            Image.Transform.AFFINE,
            (1, shear, extra if shear < 0 else 0, 0, 1, 0),
            resample=Image.Resampling.BICUBIC,
            fillcolor=(250, 249, 246),
        )
        image = image.rotate(
            -0.7 if sample.sample_id == "smoke_009" else 1.1,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(250, 249, 246),
        )
        return image.filter(ImageFilter.GaussianBlur(0.18 if sample.sample_id == "smoke_009" else 0.35))
    if sample.sample_id == "smoke_007":
        small = image.resize((max(220, image.width // 3), max(35, image.height // 3)), Image.Resampling.LANCZOS)
        from io import BytesIO

        buffer = BytesIO()
        small.save(buffer, format="JPEG", quality=24, subsampling=2)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if sample.sample_id == "smoke_008":
        image = image.rotate(2.2, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(219, 215, 204))
        overlay = Image.new("L", image.size)
        pixels = overlay.load()
        for x in range(image.width):
            shade = int(25 + 75 * x / max(1, image.width - 1))
            for y in range(image.height):
                pixels[x, y] = shade
        shadow = Image.new("RGB", image.size, (115, 106, 91))
        return Image.blend(image, Image.composite(shadow, image, overlay), 0.18)
    return image


def blank_image() -> Image.Image:
    rng = random.Random(20260714)
    image = Image.new("RGB", (640, 120), (238, 236, 231))
    draw = ImageDraw.Draw(image)
    for _ in range(24):
        x, y = rng.randrange(640), rng.randrange(120)
        radius = rng.choice((1, 1, 2))
        shade = rng.randrange(205, 231)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(shade,) * 3)
    return image.filter(ImageFilter.GaussianBlur(0.35))


def numeric_records(sample: Sample) -> list[dict[str, object]]:
    records = []
    cursor = 0
    for raw, span_type, canonical in sample.numeric_spans:
        start = sample.reference.index(raw, cursor)
        end = start + len(raw)
        cursor = end
        records.append({"start": start, "end": end, "type": span_type, "raw": raw, "canonical_value": canonical, "preserve_leading_zero": raw.startswith(("۰", "0"))})
    return records


def build(force: bool) -> None:
    missing = [path for path in (*FONTS.values(), SOURCE_SCAN) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required assets: {missing}")
    if DATA_DIR.exists() and force:
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for sample in SAMPLES:
        output = DATA_DIR / f"{sample.sample_id}.png"
        if output.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite {output}; pass --force")
        if sample.sample_id in {"smoke_006", "smoke_019"}:
            image = scanned_crop(sample.sample_id)
        elif sample.sample_id == "smoke_020":
            image = blank_image()
        else:
            image = transform(sample, render_line(sample))
        image.save(output, format="PNG", optimize=True)
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        records.append({
            "sample_id": sample.sample_id,
            "split": "smoke",
            "image_path": output.relative_to(ROOT).as_posix(),
            "sha256": digest,
            "source_document_id": "wikimedia_babak_1953_page10" if sample.origin == "real" else f"synthetic_{sample.sample_id}",
            "writer_id": None,
            "reference_raw": sample.reference,
            "description": sample.description,
            "primary_stratum": sample.primary_stratum,
            "attributes": list(sample.attributes),
            "origin": sample.origin,
            "generation_method": sample.method,
            "numeric_spans": numeric_records(sample),
            "is_blank": sample.sample_id == "smoke_020",
            "annotator_status": "needs_human_review",
            "adjudicated": False,
        })
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    print(f"Built {len(records)} smoke images and {MANIFEST_PATH.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace the generated smoke images")
    args = parser.parse_args()
    build(args.force)


if __name__ == "__main__":
    main()
