#!/usr/bin/env python3
"""sub-station — local subtitle engine and command-line interface.

The engine transcribes English or Spanish audio with mlx_whisper or NVIDIA
Parakeet (parakeet-mlx), can translate English cues to Spanish locally, validates
the generated SubRip files, and can optionally add subtitles to a new MP4 without
re-encoding video or audio.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from collections import deque
from pathlib import Path
from typing import Callable

VERSION = "2.3.1"
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
MIN_CUE_SECONDS = 0.08
LANGUAGES = {
    "en": {"label": "English", "metadata": "eng"},
    "es": {"label": "Spanish", "metadata": "spa"},
}
ENGINES = {"whisper": "Whisper", "parakeet": "Parakeet"}

# These flags prevent Whisper from feeding a failed window back into the next
# window and reduce hallucinated cues across long music or silence stretches.
# `--word-timestamps True` is required by the hallucination-silence threshold.
WHISPER_ANTI_COLLAPSE = [
    "--condition-on-previous-text", "False",
    "--word-timestamps", "True",
    "--hallucination-silence-threshold", "2",
]

# Parakeet ends a cue only at sentence punctuation, so a long unpunctuated
# stretch would become one oversized subtitle. These limits (parakeet-mlx 0.4.1+)
# keep cues near two readable lines and close them when the speaker pauses.
PARAKEET_CUE_LIMITS = [
    "--max-words", "14",
    "--max-duration", "6",
    "--silence-gap", "1.5",
]

AD_PATTERNS = [
    r"downloaded from", r"subtitles?\s+by", r"translated\s+by",
    r"corrected\s+by", r"sync(?:ed|hronized)?\s+by", r"ripped\s+by",
    r"www\.\S+", r"https?://\S+",
    r"\b[\w-]+\.(?:com|org|net|tv|io|co|me)\b",
]
# URL patterns can remove a genuine spoken website line. The accepted safeguard
# is that clean_ads always writes a separate copy and leaves the source untouched.
_AD_RE = re.compile("|".join(AD_PATTERNS), re.IGNORECASE)

# Temporary files are created owner-only (0600). Published subtitles get the
# usual umask-based mode instead, so media servers running as another user can
# read them. os.umask can only be read by setting it, so read it once at import,
# before the GUI starts any worker thread.
_UMASK = os.umask(0)
os.umask(_UMASK)
NEW_FILE_MODE = 0o666 & ~_UMASK
_TIMESTAMP_RE = re.compile(
    r"^\s*(?P<start>\d{2,}:\d{2}:\d{2},\d{3})\s+-->\s+"
    r"(?P<end>\d{2,}:\d{2}:\d{2},\d{3})(?:\s+.*)?$"
)


class EngineError(RuntimeError):
    """A pipeline failure whose message is safe to display to the user."""


class CancelledError(EngineError):
    """The user cancelled a running external process."""


LogCallback = Callable[[str], None]


# ---------------------------------------------------------------------------
# Tool discovery and execution
# ---------------------------------------------------------------------------
_search_dirs_cache: list[str] | None = None

DEDICATED_BINARIES = {
    "mlx_whisper": Path.home() / "miniconda3" / "envs" / "whisper" / "bin" / "mlx_whisper",
    # A separate environment keeps parakeet-mlx's numpy>=2.2 and librosa
    # requirements from changing the working Whisper and translator setup.
    "parakeet-mlx": Path.home() / "miniconda3" / "envs" / "parakeet" / "bin" / "parakeet-mlx",
}
TRANSLATION_PYTHON = Path.home() / "miniconda3" / "envs" / "whisper" / "bin" / "python"


def _login_shell_dirs() -> list[str]:
    """Return the login-shell PATH used by a Finder-launched application."""
    global _search_dirs_cache
    if _search_dirs_cache is None:
        dirs: list[str] = []
        try:
            shell = os.environ.get("SHELL", "/bin/zsh")
            # Keep -lic: -lc does not source ~/.zshrc, where conda init adds PATH.
            # Markers isolate PATH from prompt or startup noise printed by the rc files.
            result = subprocess.run(
                [shell, "-lic", 'printf "___PATH___%s___ENDPATH___" "$PATH"'],
                capture_output=True,
                text=True,
                timeout=8,
                stdin=subprocess.DEVNULL,
            )
            match = re.search(r"___PATH___(.*?)___ENDPATH___", result.stdout, re.S)
            if match:
                dirs = match.group(1).split(os.pathsep)
        except Exception:
            pass
        dirs += ["/opt/homebrew/bin", "/usr/local/bin"]
        _search_dirs_cache = list(dict.fromkeys(d for d in dirs if d))
    return _search_dirs_cache


def resolve_binary(name: str) -> str:
    """Return an executable's absolute path or raise an actionable error."""
    dedicated = DEDICATED_BINARIES.get(name)
    if dedicated and dedicated.is_file() and os.access(dedicated, os.X_OK):
        return str(dedicated)
    found = shutil.which(name)
    if found:
        return found
    for directory in _login_shell_dirs():
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise EngineError(
        f"'{name}' was not found. Install the required tools once:\n"
        "  mlx-whisper in ~/miniconda3/envs/whisper\n"
        "  ffmpeg with Homebrew"
    )


def check_dependencies() -> dict[str, str]:
    """Resolve every external tool required by the app."""
    return {name: resolve_binary(name) for name in ("mlx_whisper", "ffmpeg")}


def check_parakeet_dependencies() -> str:
    """Resolve the optional Parakeet transcriber or explain its one-time setup."""
    try:
        return resolve_binary("parakeet-mlx")
    except EngineError:
        raise EngineError(
            "Parakeet is not installed. Set it up once:\n"
            "  conda create -n parakeet python=3.12\n"
            "  ~/miniconda3/envs/parakeet/bin/python -m pip install 'parakeet-mlx>=0.4.1'"
        ) from None


def _translation_worker_path() -> Path:
    """Return the bundled or source-tree translation worker path."""
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    worker = bundle_root / "translation_worker.py"
    if worker.is_file():
        return worker
    raise EngineError("the bundled English-to-Spanish translation worker is missing")


def check_translation_dependencies() -> dict[str, str]:
    """Verify the local translator packages and cached model without a network."""
    if not TRANSLATION_PYTHON.is_file() or not os.access(TRANSLATION_PYTHON, os.X_OK):
        raise EngineError(
            "Spanish translation is not prepared. The Whisper environment is missing."
        )
    worker = _translation_worker_path()
    env = _tool_environment()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    try:
        result = subprocess.run(
            [str(TRANSLATION_PYTHON), str(worker), "--check"],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise EngineError(f"could not check the local Spanish translator: {error}")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise EngineError(
            "Spanish translation needs its one-time local model setup."
            + (f"\n{detail}" if detail else "")
        )
    return {"python": str(TRANSLATION_PYTHON), "model": result.stdout.strip()}


def _tool_environment() -> dict[str, str]:
    env = os.environ.copy()
    dedicated_dirs = [str(path.parent) for path in DEDICATED_BINARIES.values()]
    inherited_dirs = env.get("PATH", "").split(os.pathsep)
    dirs = list(dict.fromkeys(dedicated_dirs + _login_shell_dirs() + inherited_dirs))
    env["PATH"] = os.pathsep.join(path for path in dirs if path)
    return env


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    """Stop an external tool and any child process it launched."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=4)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _raise_if_cancelled(
    cancel_event: threading.Event | None,
    message: str,
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError(message)


def run(
    cmd: list[str],
    step: str,
    *,
    on_log: LogCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Run one tool with live logging, cancellation, and useful failure details."""
    print(f"\n== {step} ==")
    try:
        process = subprocess.Popen(
            cmd,
            env=_tool_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
    except FileNotFoundError:
        raise EngineError(f"{step} failed — '{cmd[0]}' was not found.")

    lines: queue.Queue[str | None] = queue.Queue()
    tail: deque[str] = deque(maxlen=12)

    def read_output() -> None:
        assert process.stdout is not None
        for line in iter(process.stdout.readline, ""):
            lines.put(line)
        lines.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    reader_done = False
    cancelled = False

    try:
        while process.poll() is None or not reader_done or not lines.empty():
            if cancel_event is not None and cancel_event.is_set() and process.poll() is None:
                cancelled = True
                _terminate_process_group(process)
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                reader_done = True
                continue
            clean_line = line.rstrip()
            if clean_line:
                tail.append(clean_line)
            print(line, end="")
            if on_log is not None:
                on_log(line)

        return_code = process.wait()
    except BaseException:
        # The tool runs in its own session, so a terminal Ctrl-C never reaches
        # it. Stop it here before KeyboardInterrupt (or any error) propagates.
        _terminate_process_group(process)
        raise
    reader.join(timeout=1)
    if process.stdout is not None:
        process.stdout.close()
    if cancel_event is not None and cancel_event.is_set():
        cancelled = True
    if cancelled:
        raise CancelledError(f"{step} was cancelled. No final output was written.")
    if return_code != 0:
        details = "\n".join(tail)
        suffix = f"\n\nLast tool output:\n{details}" if details else ""
        raise EngineError(f"{step} failed (exit code {return_code}).{suffix}")


# ---------------------------------------------------------------------------
# SRT parsing and cleaning
# ---------------------------------------------------------------------------
def _srt_seconds(stamp: str) -> float:
    hms, separator, milliseconds = stamp.strip().partition(",")
    if separator != "," or len(milliseconds) != 3:
        raise ValueError(f"invalid SubRip timestamp: {stamp}")
    hours, minutes, seconds = (int(value) for value in hms.split(":"))
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid SubRip timestamp: {stamp}")
    return hours * 3600 + minutes * 60 + seconds + int(milliseconds) / 1000


def _rebuild(text: str, drop_line=None) -> tuple[str, int, int]:
    """Validate and rebuild SubRip text, returning output, kept, and parsed cues."""
    stripped = text.strip()
    blocks = re.split(r"\n\s*\n", stripped) if stripped else []
    kept: list[tuple[str, str]] = []
    parsed = 0

    for block in blocks:
        lines = block.splitlines()
        timestamp_indexes = [i for i, line in enumerate(lines) if "-->" in line]
        if not timestamp_indexes:
            continue
        if len(timestamp_indexes) != 1:
            raise ValueError("a subtitle cue contains multiple timestamp lines")
        timestamp_index = timestamp_indexes[0]
        timestamp_line = lines[timestamp_index].strip()
        match = _TIMESTAMP_RE.match(timestamp_line)
        if match is None:
            raise ValueError(f"invalid SubRip timestamp line: {timestamp_line}")
        start = _srt_seconds(match.group("start"))
        end = _srt_seconds(match.group("end"))
        parsed += 1

        body = lines[timestamp_index + 1:]
        if drop_line is not None:
            body = [line for line in body if not drop_line(line)]
        cue_text = "\n".join(body).strip()
        # Cues shorter than 80 ms cannot be read and are a strong Whisper
        # artifact signal. Keep all longer cues, including lyrics over music.
        if not cue_text or end - start < MIN_CUE_SECONDS:
            continue
        kept.append((timestamp_line, cue_text))

    if stripped and parsed == 0:
        raise ValueError("no valid SubRip subtitle cues were found")

    output = "\n\n".join(
        f"{index}\n{timestamp}\n{cue_text}"
        for index, (timestamp, cue_text) in enumerate(kept, 1)
    )
    return (output + "\n" if output else ""), len(kept), parsed


def _read_srt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise EngineError(
            f"{path.name} is not valid UTF-8 text. Convert it to UTF-8 and try again."
        )
    except OSError as error:
        raise EngineError(f"could not read {path.name}: {error}")


def _apply_new_file_mode(path: Path) -> None:
    # FAT and exFAT volumes store no permission bits and may reject chmod; the
    # file is still usable there, so this is best effort.
    with contextlib.suppress(OSError):
        os.chmod(path, NEW_FILE_MODE)


def _atomic_write(path: Path, text: str) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            temp_path = Path(handle.name)
        _apply_new_file_mode(temp_path)
        temp_path.replace(path)
    except OSError as error:
        raise EngineError(f"could not write {path.name}: {error}")
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _publish_by_reserved_name(staged: Path, destination: Path) -> None:
    """Publish where hard links are unsupported, still never replacing a file.

    The destination name is reserved with an exclusive create, then the complete
    file is renamed over that empty placeholder. Readers can briefly see an empty
    file, but never a partial one, and an existing file is never touched.
    """
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise EngineError(
            f"output already exists, not overwriting: {destination.name}"
        ) from None
    except OSError as error:
        raise EngineError(f"could not publish {destination.name}: {error}") from error
    os.close(descriptor)
    try:
        os.replace(staged, destination)
    except OSError as error:
        # Remove only the empty placeholder created above.
        with contextlib.suppress(OSError):
            destination.unlink()
        raise EngineError(f"could not publish {destination.name}: {error}") from error


def _publish_new_file(staged: Path, destination: Path) -> None:
    """Publish a complete file atomically without ever replacing a destination."""
    try:
        os.link(staged, destination)
    except FileExistsError:
        raise EngineError(f"output already exists, not overwriting: {destination.name}")
    except OSError:
        # exFAT, FAT32 and many network shares have no hard links (EPERM or
        # ENOTSUP). Failing here would let the caller's cleanup delete a finished file.
        _publish_by_reserved_name(staged, destination)
        return
    try:
        staged.unlink()
    except OSError:
        # The destination already points to the complete inode. The caller's
        # temporary-directory cleanup gets a second chance to remove this name.
        pass


def _write_new_file(path: Path, text: str) -> None:
    """Write text through a same-folder temporary file and exclusive publish."""
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            temp_path = Path(handle.name)
        _apply_new_file_mode(temp_path)
        _publish_new_file(temp_path, path)
    except OSError as error:
        raise EngineError(f"could not write {path.name}: {error}")
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def clean_srt(path: Path) -> tuple[int, int]:
    """Clean a generated SRT atomically while preserving an all-invalid raw file."""
    try:
        output, kept, parsed = _rebuild(_read_srt(path))
    except ValueError as error:
        raise EngineError(f"could not validate {path.name}: {error}")
    if kept == 0:
        return 0, parsed
    _atomic_write(path, output)
    return kept, parsed - kept


def _is_ad_line(line: str) -> bool:
    return bool(_AD_RE.search(line))


def clean_ads(srt_path: Path) -> tuple[Path, int, int, int]:
    """Write a validated promo-free copy without modifying the input SRT."""
    out_path = srt_path.with_name(f"{srt_path.stem}-clean.srt")
    if out_path.exists():
        raise EngineError(f"output already exists, not overwriting: {out_path.name}")
    source = _read_srt(srt_path)
    removed_lines = sum(1 for line in source.splitlines() if _is_ad_line(line))
    try:
        output, kept, parsed = _rebuild(source, drop_line=_is_ad_line)
    except ValueError as error:
        raise EngineError(f"could not read {srt_path.name} as a SubRip file: {error}")
    _write_new_file(out_path, output)
    return out_path, kept, removed_lines, parsed - kept


def find_repetition_warnings(srt_path: Path) -> list[str]:
    """Return short phrases repeated at least three times within 15 seconds.

    Repetitions are reported for review, never deleted: real songs and chants may
    repeat intentionally.
    """
    try:
        normalized, _kept, _parsed = _rebuild(_read_srt(srt_path))
    except ValueError as error:
        raise EngineError(f"could not inspect {srt_path.name}: {error}")
    recent: dict[str, list[float]] = {}
    display: dict[str, str] = {}
    warnings: set[str] = set()
    for block in re.split(r"\n\s*\n", normalized.strip()):
        lines = block.splitlines()
        timestamp_index = next(i for i, line in enumerate(lines) if "-->" in line)
        match = _TIMESTAMP_RE.match(lines[timestamp_index])
        assert match is not None
        start = _srt_seconds(match.group("start"))
        text = " ".join(lines[timestamp_index + 1:]).strip()
        key = re.sub(r"[^\w]+", " ", text.casefold()).strip()
        if not key or len(key.split()) > 8:
            continue
        display.setdefault(key, text)
        times = [value for value in recent.get(key, []) if start - value <= 15]
        times.append(start)
        recent[key] = times
        if len(times) >= 3:
            warnings.add(display[key])
    return sorted(warnings, key=str.casefold)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------
def _transcriber_command(
    movie: Path,
    language: str,
    engine: str,
    temp_dir: Path,
) -> tuple[list[str], str]:
    """Build the speech-recognition command and its user-facing step name."""
    if engine == "parakeet":
        # Parakeet v3 detects the spoken language itself and has no language flag.
        # Format, template and model are explicit so PARAKEET_* environment
        # variables cannot change the file name this step expects.
        return [
            resolve_binary("parakeet-mlx"), str(movie),
            "--model", PARAKEET_MODEL,
            *PARAKEET_CUE_LIMITS,
            "--output-format", "srt",
            "--output-dir", str(temp_dir),
            "--output-template", "{filename}",
            "--verbose",
        ], "Transcribing audio with Parakeet (language detected automatically)"
    return [
        resolve_binary("mlx_whisper"), str(movie),
        "--model", WHISPER_MODEL,
        "--language", language,
        *WHISPER_ANTI_COLLAPSE,
        "--output-format", "srt",
        "--output-dir", str(temp_dir),
        "--output-name", movie.stem,
    ], f"Transcribing {LANGUAGES[language]['label']} audio with Whisper"


def transcribe(
    movie: Path,
    language: str = "en",
    engine: str = "whisper",
    *,
    on_log: LogCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> Path:
    """Transcribe a movie to a new SRT using an atomic final-output handoff."""
    if language not in LANGUAGES:
        raise EngineError(f"unsupported audio language: {language}")
    if engine not in ENGINES:
        raise EngineError(f"unsupported speech engine: {engine}")
    srt = movie.with_suffix(".srt")
    if srt.exists():
        raise EngineError(f"{srt.name} already exists, not overwriting. Rename it first.")

    try:
        temp_dir = Path(tempfile.mkdtemp(prefix=".sub-station-", dir=movie.parent))
    except OSError as error:
        raise EngineError(f"could not prepare an output beside {movie.name}: {error}")
    temp_srt = temp_dir / f"{movie.stem}.srt"
    try:
        command, step = _transcriber_command(movie, language, engine, temp_dir)
        run(command, step=step, on_log=on_log, cancel_event=cancel_event)
        # parakeet-mlx reports a failed file and still exits 0, so the missing
        # output is the only failure signal for that engine.
        if not temp_srt.is_file():
            raise EngineError(
                f"expected subtitle file was not created: {temp_srt.name}. "
                "The transcriber output above explains why."
            )
        if srt.exists():
            raise EngineError(f"{srt.name} appeared during processing; it was not overwritten.")
        _raise_if_cancelled(
            cancel_event,
            "Transcription was cancelled. No final SRT was written.",
        )
        _publish_new_file(temp_srt, srt)
        return srt
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def translate_srt(
    srt: Path,
    *,
    on_log: LogCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> Path:
    """Translate validated English cues to Spanish while preserving their timing."""
    output_path = srt.with_name(f"{srt.stem}-es.srt")
    if output_path.exists():
        raise EngineError(f"{output_path.name} already exists, not overwriting. Rename it first.")

    try:
        normalized, kept, _parsed = _rebuild(_read_srt(srt))
    except ValueError as error:
        raise EngineError(f"could not validate {srt.name} before translation: {error}")
    if kept == 0:
        raise EngineError(f"{srt.name} has no usable cues to translate")

    cues: list[tuple[str, str]] = []
    for block in re.split(r"\n\s*\n", normalized.strip()):
        lines = block.splitlines()
        timestamp_index = next(i for i, line in enumerate(lines) if "-->" in line)
        cues.append((lines[timestamp_index], "\n".join(lines[timestamp_index + 1:]).strip()))

    check_translation_dependencies()
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix=".sub-station-translate-", dir=srt.parent))
    except OSError as error:
        raise EngineError(f"could not prepare translation beside {srt.name}: {error}")
    input_path = temp_dir / "input.json"
    result_path = temp_dir / "output.json"
    try:
        _atomic_write(input_path, json.dumps([text for _stamp, text in cues], ensure_ascii=False))
        run(
            [
                str(TRANSLATION_PYTHON), str(_translation_worker_path()),
                "--input", str(input_path), "--output", str(result_path),
            ],
            step=f"Translating {len(cues)} subtitle cues to Spanish",
            on_log=on_log,
            cancel_event=cancel_event,
        )
        _raise_if_cancelled(
            cancel_event,
            "Translation was cancelled. The English SRT was preserved.",
        )
        try:
            translated = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EngineError(f"the translator returned an unreadable result: {error}")
        if not isinstance(translated, list) or len(translated) != len(cues):
            raise EngineError(
                "the translator returned a different number of cues; no Spanish SRT was written"
            )
        if not all(isinstance(text, str) and text.strip() for text in translated):
            raise EngineError("the translator returned an empty cue; no Spanish SRT was written")

        output = "\n\n".join(
            f"{index}\n{timestamp}\n{text.strip()}"
            for index, ((timestamp, _source), text) in enumerate(zip(cues, translated), 1)
        ) + "\n"
        if output_path.exists():
            raise EngineError(f"{output_path.name} appeared during processing; it was not overwritten.")
        _raise_if_cancelled(
            cancel_event,
            "Translation was cancelled. The English SRT was preserved.",
        )
        _write_new_file(output_path, output)
        return output_path
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def embed(
    movie: Path,
    srt: Path,
    language: str = "en",
    *,
    on_log: LogCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> Path:
    """Embed an SRT into a new MP4 through a disposable partial output."""
    if language not in LANGUAGES:
        raise EngineError(f"unsupported subtitle language: {language}")
    subbed = movie.with_name(f"{movie.stem}-subbed.mp4")
    if subbed.exists():
        raise EngineError(f"output already exists, not overwriting: {subbed.name}")

    try:
        descriptor, partial_name = tempfile.mkstemp(
            prefix=f".{movie.stem}-subbed.", suffix=".mp4", dir=movie.parent
        )
    except OSError as error:
        raise EngineError(f"could not prepare an MP4 beside {movie.name}: {error}")
    os.close(descriptor)
    partial = Path(partial_name)
    partial.unlink()
    try:
        run(
            [
                resolve_binary("ffmpeg"),
                "-i", str(movie),
                "-i", str(srt),
                # Map primary video and all audio (if any), not every source stream.
                # Mapping existing PGS/VOBSUB tracks would make mov_text conversion fail.
                "-map", "0:v:0",
                "-map", "0:a?",
                "-map", "1",
                "-c", "copy",
                "-c:s", "mov_text",
                "-metadata:s:s:0", f"language={LANGUAGES[language]['metadata']}",
                str(partial),
            ],
            step=f"Embedding subtitles into {subbed.name}",
            on_log=on_log,
            cancel_event=cancel_event,
        )
        if not partial.is_file():
            raise EngineError("ffmpeg completed without creating the expected MP4.")
        if subbed.exists():
            raise EngineError(f"{subbed.name} appeared during processing; it was not overwritten.")
        _raise_if_cancelled(
            cancel_event,
            "Embedding was cancelled. Completed SRT files were preserved.",
        )
        _publish_new_file(partial, subbed)
        return subbed
    finally:
        if partial.exists():
            partial.unlink()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a timed English or Spanish SRT from a movie's audio "
                    "with Whisper or Parakeet."
    )
    parser.add_argument("movie", help="video file")
    parser.add_argument(
        "--language",
        choices=sorted(LANGUAGES),
        default="en",
        help="spoken audio language: en (default) or es",
    )
    parser.add_argument(
        "--engine",
        choices=sorted(ENGINES),
        default="whisper",
        help="speech recognizer: whisper (default) or parakeet "
             "(NVIDIA Parakeet TDT v3; detects the spoken language itself)",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="also create a new *-subbed.mp4 without re-encoding video or audio",
    )
    parser.add_argument(
        "--translate-es",
        action="store_true",
        help="translate an English transcript to a separate *-es.srt locally",
    )
    parser.add_argument("--version", action="version", version=f"sub-station {VERSION}")
    args = parser.parse_args()

    movie = Path(args.movie).expanduser()
    if not movie.is_file():
        sys.exit(f"ERROR: file not found: {movie}")
    movie = Path(os.path.abspath(movie))

    try:
        if args.translate_es and args.language != "en":
            raise EngineError("--translate-es requires English audio (--language en)")
        srt = transcribe(movie, args.language, args.engine)
        kept, dropped = clean_srt(srt)
        if kept == 0:
            raise EngineError(
                f"no usable subtitles were produced ({dropped} invalid cues). "
                f"The raw transcript was preserved at {srt}."
            )
        print(f"OK: {srt} ({kept} subtitles, {dropped} invalid cues removed)")
        repetitions = find_repetition_warnings(srt)
        if repetitions:
            print("REVIEW: possible repeated transcription artifacts: " + ", ".join(repetitions))
        subtitle_language = args.language
        if args.translate_es:
            srt = translate_srt(srt)
            subtitle_language = "es"
            print(f"Spanish SRT: {srt}")
        if args.embed:
            subbed = embed(movie, srt, subtitle_language)
            print(f"Done: {subbed}")
    except EngineError as error:
        sys.exit(f"ERROR: {error}")
    except KeyboardInterrupt:
        print("\nCancelled. Completed files were kept; partial outputs were removed.",
              file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
