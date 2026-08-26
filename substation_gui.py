#!/usr/bin/env python3
"""sub-station 2.3.1 — macOS desktop interface for the shared subtitle engine."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# A PyInstaller --windowed app has no console. Keep Python writes safe; external
# tool output is captured by the engine and surfaced in the in-app activity log.
if sys.stdout is None or sys.stderr is None:
    _devnull = open(os.devnull, "w")
    sys.stdout = sys.stdout or _devnull
    sys.stderr = sys.stderr or _devnull

from substation import (
    VERSION,
    CancelledError,
    EngineError,
    check_dependencies,
    check_translation_dependencies,
    clean_ads,
    clean_srt,
    embed,
    find_repetition_warnings,
    transcribe,
    translate_srt,
)

NAVY = "#16213E"
DOT = "#7CB3E8"
GOLD = "#EFC15E"
SURFACE = "#F5F5F7"
MUTED = "#5F6673"

VIDEO_TYPES = [
    ("Video files", "*.mp4 *.m4v *.mov *.mkv *.avi"),
    ("All files", "*.*"),
]
SRT_TYPES = [("SubRip subtitles", "*.srt"), ("All files", "*.*")]
SUBTITLE_MODES = {
    "English audio → English SRT": ("en", "en"),
    "English audio → Spanish SRT": ("en", "es"),
    "Spanish audio → Spanish SRT": ("es", "es"),
}
EXISTING_SRT_LANGUAGES = {"Spanish": "es", "English": "en"}


class SubStation(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"sub-station {VERSION} — sanbluedot")
        self.geometry("760x610")
        self.minsize(700, 560)
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._busy = False
        self._cancel_event: threading.Event | None = None
        self._close_when_done = False
        self._last_output: Path | None = None
        self._tools_ready = False
        self._translator_ready = False

        self._configure_styles()
        self._build_header()
        self._build_body()
        self._bind_shortcuts()
        self.after(80, self._poll_events)
        threading.Thread(target=self._dependency_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        if "aqua" in style.theme_names():
            style.theme_use("aqua")
        style.configure("Title.TLabel", font=("Helvetica Neue", 17, "bold"))
        style.configure("Section.TLabel", font=("Helvetica Neue", 12, "bold"))
        style.configure("Muted.TLabel", foreground=MUTED, font=("Helvetica Neue", 11))
        style.configure("Status.TLabel", font=("Helvetica Neue", 11, "bold"))
        style.configure("Primary.TButton", font=("Helvetica Neue", 12, "bold"), padding=(16, 8))
        style.configure("Secondary.TButton", padding=(12, 7))
        style.configure("Tool.TLabel", font=("Helvetica Neue", 10))

    def _build_header(self) -> None:
        header = tk.Frame(self, bg=NAVY, height=72)
        header.pack(fill="x")
        header.pack_propagate(False)

        left = tk.Frame(header, bg=NAVY)
        left.pack(side="left", padx=22, pady=13)
        tk.Label(
            left, text="sanblue", bg=NAVY, fg=GOLD,
            font=("Helvetica Neue", 19, "bold"),
        ).pack(side="left")
        tk.Label(
            left, text="dot", bg=NAVY, fg=DOT,
            font=("Helvetica Neue", 10, "bold"),
        ).pack(side="left", pady=(0, 10))
        tk.Label(
            left, text="  —  sub-station", bg=NAVY, fg="white",
            font=("Helvetica Neue", 15),
        ).pack(side="left")

        self.tools_label = tk.Label(
            header, text="Checking tools…", bg=NAVY, fg="white",
            font=("Helvetica Neue", 10),
        )
        self.tools_label.pack(side="right", padx=22)

    def _build_body(self) -> None:
        container = ttk.Frame(self, padding=(18, 14, 18, 16))
        container.pack(fill="both", expand=True)

        notebook = ttk.Notebook(container)
        notebook.pack(fill="both", expand=True)
        notebook.add(self._build_generate_tab(notebook), text="  Generate  ")
        notebook.add(self._build_clean_tab(notebook), text="  Clean SRT  ")

    def _build_generate_tab(self, parent: ttk.Notebook) -> ttk.Frame:
        tab = ttk.Frame(parent, padding=20)
        tab.columnconfigure(0, weight=1)

        ttk.Label(tab, text="Generate subtitles", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            tab,
            text="Choose a movie and create subtitles in English or Spanish.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 16))

        file_group = ttk.LabelFrame(tab, text="Movie", padding=12)
        file_group.grid(row=2, column=0, sticky="ew")
        file_group.columnconfigure(0, weight=1)
        self.movie_var = tk.StringVar()
        self.movie_entry = ttk.Entry(file_group, textvariable=self.movie_var)
        self.movie_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(
            file_group, text="Choose…", style="Secondary.TButton",
            command=self._pick_movie,
        ).grid(row=0, column=1)

        options = ttk.Frame(tab)
        options.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        options.columnconfigure(1, weight=1)
        ttk.Label(options, text="Subtitle mode:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.language_var = tk.StringVar(value="English audio → English SRT")
        self.language_box = ttk.Combobox(
            options,
            textvariable=self.language_var,
            values=list(SUBTITLE_MODES),
            state="readonly",
            width=32,
        )
        self.language_box.grid(row=0, column=1, sticky="w")
        self.embed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options,
            text="Also create a new MP4 with the subtitle track (no re-encoding)",
            variable=self.embed_var,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Label(
            options,
            text="English → Spanish is machine-translated locally after transcription; timing is preserved.",
            style="Muted.TLabel",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(options, text="Existing SRT language:").grid(
            row=3, column=0, sticky="w", padx=(0, 10), pady=(10, 0)
        )
        self.existing_language_var = tk.StringVar(value="Spanish")
        self.existing_language_box = ttk.Combobox(
            options,
            textvariable=self.existing_language_var,
            values=list(EXISTING_SRT_LANGUAGES),
            state="readonly",
            width=14,
        )
        self.existing_language_box.grid(row=3, column=1, sticky="w", pady=(10, 0))

        actions = ttk.Frame(tab)
        actions.grid(row=4, column=0, sticky="ew", pady=(16, 10))
        self.generate_button = ttk.Button(
            actions, text="Generate subtitles", style="Primary.TButton",
            command=self._start_generate,
        )
        self.generate_button.pack(side="left")
        self.embed_existing_button = ttk.Button(
            actions, text="Embed existing SRT…", style="Secondary.TButton",
            command=self._start_embed_existing,
        )
        self.embed_existing_button.pack(side="left", padx=(10, 0))
        self.cancel_button = ttk.Button(
            actions, text="Cancel", style="Secondary.TButton",
            command=self._cancel, state="disabled",
        )
        self.cancel_button.pack(side="right")

        self.progress = ttk.Progressbar(tab, mode="indeterminate")
        self.progress.grid(row=5, column=0, sticky="ew", pady=(0, 7))
        status_row = ttk.Frame(tab)
        status_row.grid(row=6, column=0, sticky="ew")
        status_row.columnconfigure(0, weight=1)
        self.status_label = ttk.Label(status_row, text="Ready.", style="Status.TLabel")
        self.status_label.grid(row=0, column=0, sticky="w")
        self.new_job_button = ttk.Button(
            status_row, text="Start New Job", command=self._reset_job,
            state="disabled",
        )
        self.new_job_button.grid(row=0, column=1, padx=(8, 0))
        self.reveal_button = ttk.Button(
            status_row, text="Reveal in Finder", command=self._reveal_output,
            state="disabled",
        )
        self.reveal_button.grid(row=0, column=2, padx=(8, 0))

        log_group = ttk.LabelFrame(tab, text="Activity", padding=8)
        log_group.grid(row=7, column=0, sticky="nsew", pady=(12, 0))
        tab.rowconfigure(7, weight=1, minsize=155)
        log_group.rowconfigure(0, weight=1)
        log_group.columnconfigure(0, weight=1)
        self.log_text = tk.Text(
            log_group,
            height=7,
            wrap="word",
            background=SURFACE,
            foreground=NAVY,
            relief="flat",
            font=("Menlo", 10),
            padx=8,
            pady=8,
            state="disabled",
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_group, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)
        return tab

    def _build_clean_tab(self, parent: ttk.Notebook) -> ttk.Frame:
        tab = ttk.Frame(parent, padding=20)
        tab.columnconfigure(0, weight=1)

        ttk.Label(tab, text="Clean a downloaded SRT", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            tab,
            text="Remove known promo and credit lines while preserving the original file.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 18))

        file_group = ttk.LabelFrame(tab, text="Subtitle file", padding=12)
        file_group.grid(row=2, column=0, sticky="ew")
        file_group.columnconfigure(0, weight=1)
        self.srt_var = tk.StringVar()
        ttk.Entry(file_group, textvariable=self.srt_var).grid(
            row=0, column=0, sticky="ew", padx=(0, 8)
        )
        ttk.Button(
            file_group, text="Choose…", style="Secondary.TButton",
            command=self._pick_srt,
        ).grid(row=0, column=1)

        ttk.Label(
            tab,
            text="The cleaned copy is saved as *-clean.srt. Invalid SRT files are rejected; "
                 "the source is never changed.",
            style="Muted.TLabel",
            wraplength=620,
            justify="left",
        ).grid(row=3, column=0, sticky="w", pady=(14, 16))

        clean_actions = ttk.Frame(tab)
        clean_actions.grid(row=4, column=0, sticky="ew")
        self.clean_button = ttk.Button(
            clean_actions, text="Clean SRT", style="Primary.TButton",
            command=self._clean_srt,
        )
        self.clean_button.pack(side="left")
        self.clean_reveal_button = ttk.Button(
            clean_actions, text="Reveal in Finder", command=self._reveal_output,
            state="disabled",
        )
        self.clean_reveal_button.pack(side="left", padx=(10, 0))

        self.clean_status = ttk.Label(tab, text="Ready.", style="Status.TLabel")
        self.clean_status.grid(row=5, column=0, sticky="w", pady=(20, 0))
        return tab

    def _bind_shortcuts(self) -> None:
        self.bind("<Command-o>", lambda _event: self._pick_movie())
        self.bind("<Command-Return>", lambda _event: self._start_generate())
        self.bind("<Escape>", lambda _event: self._cancel())

    # ------------------------------------------------------------------
    # File selection and dependency status
    # ------------------------------------------------------------------
    def _pick_movie(self) -> None:
        path = filedialog.askopenfilename(title="Choose a movie", filetypes=VIDEO_TYPES)
        if path:
            self.movie_var.set(path)

    def _pick_srt(self) -> None:
        path = filedialog.askopenfilename(title="Choose an SRT", filetypes=SRT_TYPES)
        if path:
            self.srt_var.set(path)

    def _dependency_worker(self) -> None:
        try:
            tools = check_dependencies()
            try:
                translator = check_translation_dependencies()
                translation_status: tuple[bool, object] = (True, translator)
            except EngineError as error:
                translation_status = (False, str(error))
            self._events.put(("dependencies", (True, tools, translation_status)))
        except EngineError as error:
            self._events.put(("dependencies", (False, str(error), (False, "Core tools missing"))))

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------
    def _selected_mode(self) -> tuple[str, str]:
        return SUBTITLE_MODES[self.language_var.get()]

    def _valid_movie(self) -> Path | None:
        raw = self.movie_var.get().strip()
        movie = Path(raw).expanduser()
        if not raw or not movie.is_file():
            messagebox.showwarning("Movie required", "Choose a valid movie file first.")
            return None
        if not os.access(movie.parent, os.W_OK):
            messagebox.showerror(
                "Folder is read-only",
                "sub-station needs permission to write the SRT next to the movie.",
            )
            return None
        return Path(os.path.abspath(movie))

    def _begin_job(self, status: str) -> bool:
        if self._busy:
            return False
        if not self._tools_ready:
            messagebox.showerror(
                "Tools unavailable",
                "mlx_whisper and ffmpeg must both be ready before processing.",
            )
            return False
        self._busy = True
        self._cancel_event = threading.Event()
        self.generate_button.configure(state="disabled")
        self.embed_existing_button.configure(state="disabled")
        self.clean_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.reveal_button.configure(state="disabled")
        self.new_job_button.configure(state="disabled")
        self.progress.stop()
        self.progress.configure(value=0)
        self.progress.start(12)
        self.status_label.configure(text=status)
        self._clear_log()
        return True

    def _start_generate(self) -> None:
        movie = self._valid_movie()
        source_language, subtitle_language = self._selected_mode()
        if subtitle_language != source_language and not self._translator_ready:
            messagebox.showerror(
                "Spanish translation unavailable",
                "The one-time local Spanish translation setup is not ready. "
                "English and Spanish transcription modes are still available.",
            )
            return
        if movie is None or not self._begin_job("Preparing transcription…"):
            return
        do_embed = self.embed_var.get()
        threading.Thread(
            target=self._generate_worker,
            args=(movie, source_language, subtitle_language, do_embed, self._cancel_event),
            daemon=True,
        ).start()

    def _generate_worker(
        self,
        movie: Path,
        source_language: str,
        subtitle_language: str,
        do_embed: bool,
        cancel_event: threading.Event,
    ) -> None:
        srt: Path | None = None
        output: Path | None = None
        try:
            self._events.put(("stage", "Transcribing audio…"))
            srt = transcribe(
                movie,
                source_language,
                on_log=self._queue_log,
                cancel_event=cancel_event,
            )
            output = srt
            if cancel_event.is_set():
                raise CancelledError("Transcription was cancelled. The raw SRT was preserved.")
            self._events.put(("stage", "Validating and cleaning subtitle cues…"))
            kept, dropped = clean_srt(srt)
            if kept == 0:
                raise EngineError(
                    f"No usable subtitle cues remained. The raw transcript was preserved at {srt}."
                )
            message = f"SRT ready: {srt.name}\n{kept} subtitles; {dropped} invalid cues removed."
            repetitions = find_repetition_warnings(srt)
            if repetitions:
                review = ", ".join(repetitions[:4])
                suffix = "…" if len(repetitions) > 4 else ""
                warning = (
                    "Review repeated phrases (they may be dialogue, lyrics, or Whisper artifacts): "
                    f"{review}{suffix}"
                )
                message += f"\n{warning}"
                self._events.put(("log", f"\n{warning}\n"))
            if subtitle_language != source_language:
                if cancel_event.is_set():
                    raise CancelledError("Translation was cancelled. The English SRT was preserved.")
                self._events.put(("stage", "Translating English subtitles to Spanish…"))
                translated_srt = translate_srt(
                    srt,
                    on_log=self._queue_log,
                    cancel_event=cancel_event,
                )
                message += f"\nSpanish SRT ready: {translated_srt.name}"
                output = translated_srt
            if do_embed:
                if cancel_event.is_set():
                    raise CancelledError("Embedding was cancelled. The completed SRT was preserved.")
                self._events.put(("stage", "Embedding subtitle track…"))
                try:
                    output = embed(
                        movie,
                        output,
                        subtitle_language,
                        on_log=self._queue_log,
                        cancel_event=cancel_event,
                    )
                    message += f"\nMP4 ready: {output.name}"
                except CancelledError:
                    raise
                except EngineError as error:
                    self._events.put(("done", {
                        "state": "partial",
                        "message": f"{message}\n\nEmbedding failed:\n{error}",
                        "output": output,
                    }))
                    return
            self._events.put(("done", {
                "state": "success", "message": message, "output": output,
            }))
        except CancelledError as error:
            recovered = output if output is not None and output.exists() else srt
            message = str(error)
            if recovered is not None and recovered.exists():
                message += f"\nPreserved output: {recovered}"
            self._events.put(("done", {
                "state": "cancelled", "message": message, "output": recovered,
            }))
        except EngineError as error:
            recovered = output if output is not None and output.exists() else srt
            state = "partial" if recovered is not None and recovered.exists() else "failed"
            message = str(error)
            if state == "partial":
                message = f"A completed output was preserved at {recovered}.\n\n{message}"
            self._events.put(("done", {
                "state": state, "message": message, "output": recovered,
            }))
        except Exception as error:
            recovered = output if output is not None and output.exists() else srt
            self._events.put(("done", {
                "state": "failed",
                "message": f"Unexpected error: {error}",
                "output": recovered,
            }))

    def _start_embed_existing(self) -> None:
        movie = self._valid_movie()
        if movie is None:
            return
        srt_name = filedialog.askopenfilename(title="Choose an existing SRT", filetypes=SRT_TYPES)
        if not srt_name:
            return
        srt = Path(srt_name)
        if not self._begin_job("Preparing subtitle embed…"):
            return
        threading.Thread(
            target=self._embed_worker,
            args=(
                movie,
                srt,
                EXISTING_SRT_LANGUAGES[self.existing_language_var.get()],
                self._cancel_event,
            ),
            daemon=True,
        ).start()

    def _embed_worker(
        self,
        movie: Path,
        srt: Path,
        language: str,
        cancel_event: threading.Event,
    ) -> None:
        try:
            self._events.put(("stage", "Embedding existing subtitle track…"))
            output = embed(
                movie,
                srt,
                language,
                on_log=self._queue_log,
                cancel_event=cancel_event,
            )
            self._events.put(("done", {
                "state": "success",
                "message": f"MP4 ready: {output.name}",
                "output": output,
            }))
        except CancelledError as error:
            self._events.put(("done", {
                "state": "cancelled", "message": str(error), "output": None,
            }))
        except EngineError as error:
            self._events.put(("done", {
                "state": "failed", "message": str(error), "output": None,
            }))
        except Exception as error:
            self._events.put(("done", {
                "state": "failed", "message": f"Unexpected error: {error}", "output": None,
            }))

    def _clean_srt(self) -> None:
        raw = self.srt_var.get().strip()
        srt = Path(raw).expanduser()
        if not raw or not srt.is_file():
            messagebox.showwarning("SRT required", "Choose a valid .srt file first.")
            return
        try:
            output, kept, removed_lines, dropped_cues = clean_ads(srt)
            self._last_output = output
            self.clean_reveal_button.configure(state="normal")
            self.new_job_button.configure(state="normal")
            self.clean_status.configure(
                text=(
                    f"Saved {output.name}: {kept} kept, {removed_lines} promo lines and "
                    f"{dropped_cues} invalid/empty cues removed."
                )
            )
            messagebox.showinfo(
                "Clean SRT",
                f"Cleaned copy saved:\n{output}\n\n"
                f"{kept} subtitles kept; {removed_lines} promotional lines and "
                f"{dropped_cues} invalid or empty cues removed.\n"
                "The original file is unchanged.",
            )
        except EngineError as error:
            self.new_job_button.configure(state="normal")
            self.clean_status.configure(text="Could not clean the selected file.")
            messagebox.showerror("Clean SRT", str(error))
        except Exception as error:
            self.new_job_button.configure(state="normal")
            self.clean_status.configure(text="Could not clean the selected file.")
            messagebox.showerror("Clean SRT", f"Unexpected error: {error}")

    def _cancel(self) -> None:
        if self._busy and self._cancel_event is not None:
            self._cancel_event.set()
            self.cancel_button.configure(state="disabled")
            self.status_label.configure(text="Cancelling safely…")

    def _reset_job(self) -> None:
        """Return the interface to a clean idle state without deleting outputs."""
        if self._busy:
            return
        self.movie_var.set("")
        self.srt_var.set("")
        self.language_var.set("English audio → English SRT")
        self.existing_language_var.set("Spanish")
        self.embed_var.set(False)
        self._last_output = None
        self._close_when_done = False
        self.progress.stop()
        self.progress.configure(value=0)
        self.status_label.configure(text="Ready.")
        self.clean_status.configure(text="Ready.")
        self.reveal_button.configure(state="disabled")
        self.clean_reveal_button.configure(state="disabled")
        self.new_job_button.configure(state="disabled")
        self._clear_log()
        self.movie_entry.focus_set()

    # ------------------------------------------------------------------
    # Main-thread event handling
    # ------------------------------------------------------------------
    def _queue_log(self, line: str) -> None:
        self._events.put(("log", line))

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self._events.get_nowait()
                if event == "dependencies":
                    ready, detail, translation_status = payload
                    self._tools_ready = bool(ready)
                    if ready:
                        translation_ready, translation_detail = translation_status
                        self._translator_ready = bool(translation_ready)
                        label = "● Core + Spanish ready" if translation_ready else "● Core ready"
                        self.tools_label.configure(text=label, fg=DOT)
                        tools = detail
                        self._append_log(
                            f"mlx_whisper: {tools['mlx_whisper']}\nffmpeg: {tools['ffmpeg']}\n"
                        )
                        if translation_ready:
                            self._append_log(
                                f"Spanish translator: {translation_detail['model']}\n"
                            )
                        else:
                            self._append_log(
                                f"Spanish translator not ready:\n{translation_detail}\n"
                            )
                    else:
                        self.tools_label.configure(text="● Tools missing", fg="#FF8A80")
                        self._append_log(f"Dependency check failed:\n{detail}\n")
                elif event == "stage":
                    self.status_label.configure(text=str(payload))
                    self._append_log(f"\n{payload}\n")
                elif event == "log":
                    self._append_log(str(payload))
                elif event == "done":
                    self._finish_job(payload)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(80, self._poll_events)

    def _finish_job(self, result: dict[str, object]) -> None:
        self.progress.stop()
        self.progress.configure(value=0)
        self._busy = False
        self._cancel_event = None
        self.generate_button.configure(state="normal")
        self.embed_existing_button.configure(state="normal")
        self.clean_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.new_job_button.configure(state="normal")

        state = str(result["state"])
        message = str(result["message"])
        output = result.get("output")
        if isinstance(output, Path) and output.exists():
            self._last_output = output
            self.reveal_button.configure(state="normal")

        labels = {
            "success": "Complete.",
            "partial": "Partially complete — output preserved.",
            "cancelled": "Cancelled safely.",
            "failed": "Could not finish.",
        }
        self.status_label.configure(text=labels[state])
        self._append_log(f"\n{labels[state]}\n{message}\n")

        if self._close_when_done:
            self.destroy()
            return
        if state == "success":
            messagebox.showinfo(
                "sub-station complete",
                "Finished successfully. Details and filenames are available in Activity.",
            )
        elif state in {"partial", "cancelled"}:
            messagebox.showwarning("sub-station", message)
        else:
            messagebox.showerror("sub-station", message)

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _reveal_output(self) -> None:
        if self._last_output is not None and self._last_output.exists():
            subprocess.run(["open", "-R", str(self._last_output)], check=False)

    def _on_close(self) -> None:
        if not self._busy:
            self.destroy()
            return
        if messagebox.askyesno(
            "Cancel and quit?",
            "A subtitle job is running. Cancel it safely and close sub-station?",
        ):
            self._close_when_done = True
            self._cancel()


if __name__ == "__main__":
    SubStation().mainloop()
