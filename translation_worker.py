#!/usr/bin/env python3
"""Offline English-to-Spanish translation worker for sub-station.

This script runs in the dedicated Whisper conda environment so the macOS app
does not need to bundle PyTorch and Transformers. Movie dialogue is exchanged
through temporary local JSON files and never sent to a remote service.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

MODEL = "Helsinki-NLP/opus-mt-en-es"
MAX_SOURCE_TOKENS = 512
ABBREVIATIONS = {"dr.", "mr.", "mrs.", "ms.", "st.", "vs.", "etc."}
LATAM_REPLACEMENTS = {
    "ordenadores": "computadores",
    "ordenador": "computador",
    "zumos": "jugos",
    "zumo": "jugo",
    "patatas": "papas",
    "patata": "papa",
}
NEUTRAL_PLURAL_REPLACEMENTS = {
    "vosotros": "ustedes",
    "vosotras": "ustedes",
    "vuestros": "sus",
    "vuestras": "sus",
    "vuestro": "su",
    "vuestra": "su",
    "sois": "son",
    "estáis": "están",
    "tenéis": "tienen",
    "habéis": "han",
    "podéis": "pueden",
    "queréis": "quieren",
    "debéis": "deben",
    "hacéis": "hacen",
    "decís": "dicen",
    "vais": "van",
    "venís": "vienen",
    "contadnos": "cuéntennos",
}


def cached_model_path() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=MODEL, local_files_only=True)


def download_model() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=MODEL)


def split_dialogue(text: str) -> list[str]:
    """Split multi-sentence cues so the model cannot silently omit a sentence."""
    candidates = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    fragments: list[str] = []
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        if fragments:
            previous = fragments[-1].casefold()
            compact = re.sub(r"\s+", "", fragments[-1])
            if previous in ABBREVIATIONS or re.fullmatch(r"(?:[a-zA-Z]\.){2,}", compact):
                fragments[-1] = f"{fragments[-1]} {candidate}"
                continue
        fragments.append(candidate)
    return fragments or [text.strip()]


def _case_like(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


def normalize_latam(text: str) -> str:
    """Replace unambiguous Spain usage without changing context-dependent words."""
    replacements = {**LATAM_REPLACEMENTS, **NEUTRAL_PLURAL_REPLACEMENTS}
    pattern = re.compile(
        r"\b(" + "|".join(sorted(map(re.escape, replacements), key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    result = pattern.sub(
        lambda match: _case_like(match.group(0), replacements[match.group(0).casefold()]),
        text,
    )
    # "Vale" is a Spain discourse marker only at the start of a sentence and
    # before punctuation. Do not touch the verb in phrases such as "no vale".
    return re.sub(
        r"(^|(?<=[.!?]\s))vale(?=\s*[,!.?])",
        lambda match: _case_like(match.group(0), "bien"),
        result,
        flags=re.IGNORECASE,
    )


def normalize_translation(source: str, translated: str) -> str:
    """Apply conservative source-aware terminology after regional normalization."""
    source_key = source.casefold()
    result = translated
    if re.search(r"\bcount us down\b", source_key):
        result = re.sub(
            r"\b(?:contadnos|cuéntennos)\b",
            "hagan la cuenta regresiva",
            result,
            flags=re.IGNORECASE,
        )
    if re.search(r"\b(?:cell ?phone|mobile phone|smartphone|phone)s?\b", source_key):
        result = re.sub(r"\bmóviles\b", "celulares", result, flags=re.IGNORECASE)
        result = re.sub(r"\bmóvil\b", "celular", result, flags=re.IGNORECASE)
    if re.search(r"\b(?:car|automobile|vehicle)s?\b", source_key):
        result = re.sub(r"\bcoches\b", "autos", result, flags=re.IGNORECASE)
        result = re.sub(r"\bcoche\b", "auto", result, flags=re.IGNORECASE)
    if re.search(r"\b(?:stroller|pram|baby carriage)s?\b", source_key):
        result = re.sub(r"\bcoches\b", "cochecitos", result, flags=re.IGNORECASE)
        result = re.sub(r"\bcoche\b", "cochecito", result, flags=re.IGNORECASE)
    result = normalize_latam(result)
    if "black hole" in source_key and "matter" in source_key:
        result = re.sub(r"\b(?:todo el|toda la) asunto\b", "toda la materia", result, flags=re.I)
        result = re.sub(r"\b(?:chupar|succionar)\b", "absorber", result, flags=re.I)
    return result


def _translate_fragments(
    lines: list[str],
    *,
    device: str,
    batch_size: int,
) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_path = cached_model_path()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

    model.to(device)
    model.eval()
    translated: list[str] = []

    for start in range(0, len(lines), batch_size):
        batch = lines[start:start + batch_size]
        tokenized = tokenizer(batch, padding=False, truncation=False)
        oversized = [
            start + index + 1
            for index, token_ids in enumerate(tokenized["input_ids"])
            if len(token_ids) > MAX_SOURCE_TOKENS
        ]
        if oversized:
            numbers = ", ".join(str(number) for number in oversized)
            raise ValueError(
                f"source subtitle fragment exceeds {MAX_SOURCE_TOKENS} tokens: {numbers}"
            )
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(device)
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=512,
                num_beams=4,
                early_stopping=True,
            )
        translated.extend(
            text.strip() for text in tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        )
        print(
            f"Translated {min(start + batch_size, len(lines))}/{len(lines)} fragments on {device}",
            flush=True,
        )
    return translated


def translate(lines: list[str], *, batch_size: int = 16) -> list[str]:
    import torch

    groups = [split_dialogue(line) for line in lines]
    fragments = [fragment for group in groups for fragment in group]
    if torch.backends.mps.is_available():
        try:
            translated_fragments = _translate_fragments(
                fragments,
                device="mps",
                batch_size=batch_size,
            )
        except RuntimeError as error:
            print(f"Metal translation failed; retrying safely on CPU: {error}", flush=True)
            try:
                torch.mps.empty_cache()
            except RuntimeError:
                pass
            translated_fragments = _translate_fragments(
                fragments,
                device="cpu",
                batch_size=batch_size,
            )
    else:
        translated_fragments = _translate_fragments(
            fragments,
            device="cpu",
            batch_size=batch_size,
        )

    result: list[str] = []
    cursor = 0
    for group in groups:
        translated_group = translated_fragments[cursor:cursor + len(group)]
        if len(translated_group) != len(group) or any(not item.strip() for item in translated_group):
            raise ValueError("the translator omitted a subtitle sentence")
        source_context = " ".join(group)
        normalized_group = [
            normalize_translation(source_context, translated_fragment)
            for translated_fragment in translated_group
        ]
        result.append(" ".join(normalized_group))
        cursor += len(group)
    if cursor != len(translated_fragments):
        raise ValueError("the translator returned an unexpected number of sentence fragments")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.download:
        print(download_model())
        return
    if args.check:
        import sacremoses  # noqa: F401
        import sentencepiece  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401

        print(cached_model_path())
        return
    if args.input is None or args.output is None:
        parser.error("--input and --output are required for translation")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        source = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        sys.exit(f"Could not read translation input: {error}")
    if not isinstance(source, list) or not all(isinstance(item, str) for item in source):
        sys.exit("Translation input must be a JSON array of strings.")

    try:
        result = translate(source)
    except (RuntimeError, ValueError) as error:
        sys.exit(f"Translation failed safely: {error}")
    try:
        args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    except OSError as error:
        sys.exit(f"Could not write translation output: {error}")


if __name__ == "__main__":
    main()
