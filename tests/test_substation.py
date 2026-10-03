from __future__ import annotations

import queue
import shutil
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import substation
import substation_gui
import translation_worker


class TemporaryDirectoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="substation-tests-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class SrtValidationTests(TemporaryDirectoryTest):
    def test_rebuild_accepts_crlf_and_drops_zero_duration(self) -> None:
        source = (
            "1\r\n00:00:01,000 --> 00:00:02,000\r\nHello\r\n\r\n"
            "2\r\n00:00:03,000 --> 00:00:03,000\r\nJunk\r\n"
        )
        output, kept, parsed = substation._rebuild(source)
        self.assertEqual((kept, parsed), (1, 2))
        self.assertIn("Hello", output)
        self.assertNotIn("Junk", output)

    def test_rebuild_drops_impossible_duration_but_keeps_short_speech(self) -> None:
        source = (
            "1\n00:00:01,000 --> 00:00:01,020\nThank you.\n\n"
            "2\n00:00:02,000 --> 00:00:02,160\nNo.\n"
        )
        output, kept, parsed = substation._rebuild(source)
        self.assertEqual((kept, parsed), (1, 2))
        self.assertNotIn("Thank you", output)
        self.assertIn("No.", output)

    def test_invalid_srt_is_rejected_without_output(self) -> None:
        source = self.temp_dir / "broken.srt"
        source.write_text("This is not an SRT file.\n", encoding="utf-8")
        with self.assertRaisesRegex(substation.EngineError, "no valid SubRip"):
            substation.clean_ads(source)
        self.assertFalse((self.temp_dir / "broken-clean.srt").exists())

    def test_malformed_timestamp_is_rejected(self) -> None:
        source = self.temp_dir / "broken.srt"
        source.write_text(
            "1\n00:00:01 --> 00:00:02\nHello\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(substation.EngineError, "invalid SubRip timestamp"):
            substation.clean_ads(source)

    def test_all_invalid_generated_cues_preserve_raw_transcript(self) -> None:
        source = self.temp_dir / "raw.srt"
        raw = "1\n00:00:01,000 --> 00:00:01,000\nRaw model output\n"
        source.write_text(raw, encoding="utf-8")
        kept, dropped = substation.clean_srt(source)
        self.assertEqual((kept, dropped), (0, 1))
        self.assertEqual(source.read_text(encoding="utf-8"), raw)

    def test_ad_cleaner_preserves_source_and_counts_removed_lines(self) -> None:
        source = self.temp_dir / "downloaded.srt"
        raw = (
            "1\n00:00:01,000 --> 00:00:02,000\nVisit www.example.com\nReal dialogue\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nKeep me\n"
        )
        source.write_text(raw, encoding="utf-8")
        output, kept, removed_lines, dropped_cues = substation.clean_ads(source)
        self.assertEqual((kept, removed_lines, dropped_cues), (2, 1, 0))
        self.assertEqual(source.read_text(encoding="utf-8"), raw)
        self.assertNotIn("www.example.com", output.read_text(encoding="utf-8"))
        self.assertIn("Real dialogue", output.read_text(encoding="utf-8"))

    def test_ad_cleaner_reports_invalid_cues_separately(self) -> None:
        source = self.temp_dir / "downloaded.srt"
        source.write_text(
            "1\n00:00:01,000 --> 00:00:01,020\nArtifact\n\n"
            "2\n00:00:02,000 --> 00:00:03,000\nKeep\n",
            encoding="utf-8",
        )
        output, kept, removed_lines, dropped_cues = substation.clean_ads(source)
        self.assertEqual((kept, removed_lines, dropped_cues), (1, 0, 1))
        self.assertNotIn("Artifact", output.read_text(encoding="utf-8"))

    def test_cleaner_refuses_to_overwrite_existing_output(self) -> None:
        source = self.temp_dir / "downloaded.srt"
        output = self.temp_dir / "downloaded-clean.srt"
        source.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nHello\n",
            encoding="utf-8",
        )
        output.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(substation.EngineError, "not overwriting"):
            substation.clean_ads(source)
        self.assertEqual(output.read_text(encoding="utf-8"), "keep")

    def test_repetition_warning_reports_burst_without_deleting_it(self) -> None:
        source = self.temp_dir / "repeated.srt"
        source.write_text(
            "1\n00:00:01,000 --> 00:00:01,500\nThank you.\n\n"
            "2\n00:00:02,000 --> 00:00:02,500\nThank you.\n\n"
            "3\n00:00:03,000 --> 00:00:03,500\nThank you.\n",
            encoding="utf-8",
        )
        self.assertEqual(substation.find_repetition_warnings(source), ["Thank you."])
        kept, dropped = substation.clean_srt(source)
        self.assertEqual((kept, dropped), (3, 0))


class PipelineSafetyTests(TemporaryDirectoryTest):
    def setUp(self) -> None:
        super().setUp()
        self.movie = self.temp_dir / "movie.mkv"
        self.movie.write_bytes(b"movie")
        self.srt = self.temp_dir / "movie.srt"

    def test_transcribe_uses_selected_language_and_atomic_handoff(self) -> None:
        captured: list[str] = []

        def fake_run(command, _step=None, **_kwargs) -> None:
            captured.extend(command)
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_name = command[command.index("--output-name") + 1]
            (output_dir / f"{output_name}.srt").write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nHola\n",
                encoding="utf-8",
            )

        with patch.object(substation, "resolve_binary", return_value="mlx_whisper"), patch.object(
            substation, "run", side_effect=fake_run
        ):
            result = substation.transcribe(self.movie, "es")

        self.assertEqual(result, self.srt)
        self.assertEqual(self.srt.read_text(encoding="utf-8").splitlines()[-1], "Hola")
        self.assertEqual(captured[captured.index("--language") + 1], "es")
        self.assertFalse(any(path.name.startswith(".sub-station-") for path in self.temp_dir.iterdir()))

    def test_failed_transcription_removes_partial_output(self) -> None:
        def fake_run(command, _step=None, **_kwargs) -> None:
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_name = command[command.index("--output-name") + 1]
            (output_dir / f"{output_name}.srt").write_text("partial", encoding="utf-8")
            raise substation.EngineError("simulated failure")

        with patch.object(substation, "resolve_binary", return_value="mlx_whisper"), patch.object(
            substation, "run", side_effect=fake_run
        ):
            with self.assertRaisesRegex(substation.EngineError, "simulated failure"):
                substation.transcribe(self.movie)

        self.assertFalse(self.srt.exists())
        self.assertFalse(any(path.name.startswith(".sub-station-") for path in self.temp_dir.iterdir()))

    def test_late_transcription_cancel_does_not_publish(self) -> None:
        cancel = threading.Event()

        def fake_run(command, _step=None, **_kwargs) -> None:
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_name = command[command.index("--output-name") + 1]
            (output_dir / f"{output_name}.srt").write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nHello\n",
                encoding="utf-8",
            )
            cancel.set()

        with patch.object(substation, "resolve_binary", return_value="mlx_whisper"), patch.object(
            substation, "run", side_effect=fake_run
        ):
            with self.assertRaises(substation.CancelledError):
                substation.transcribe(self.movie, cancel_event=cancel)

        self.assertFalse(self.srt.exists())

    def test_failed_embed_removes_partial_mp4(self) -> None:
        self.srt.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nHello\n",
            encoding="utf-8",
        )

        def fake_run(command, _step=None, **_kwargs) -> None:
            Path(command[-1]).write_bytes(b"partial")
            raise substation.EngineError("simulated ffmpeg failure")

        with patch.object(substation, "resolve_binary", return_value="ffmpeg"), patch.object(
            substation, "run", side_effect=fake_run
        ):
            with self.assertRaisesRegex(substation.EngineError, "simulated ffmpeg failure"):
                substation.embed(self.movie, self.srt)

        self.assertFalse((self.temp_dir / "movie-subbed.mp4").exists())
        self.assertFalse(any("-subbed." in path.name for path in self.temp_dir.iterdir()))

    def test_late_embed_cancel_does_not_publish(self) -> None:
        self.srt.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nHello\n",
            encoding="utf-8",
        )
        cancel = threading.Event()

        def fake_run(command, _step=None, **_kwargs) -> None:
            Path(command[-1]).write_bytes(b"complete but cancelled")
            cancel.set()

        with patch.object(substation, "resolve_binary", return_value="ffmpeg"), patch.object(
            substation, "run", side_effect=fake_run
        ):
            with self.assertRaises(substation.CancelledError):
                substation.embed(self.movie, self.srt, cancel_event=cancel)

        self.assertFalse((self.temp_dir / "movie-subbed.mp4").exists())

    def test_embed_sets_spanish_track_metadata(self) -> None:
        self.srt.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nHola\n",
            encoding="utf-8",
        )
        captured: list[str] = []

        def fake_run(command, _step=None, **_kwargs) -> None:
            captured.extend(command)
            Path(command[-1]).write_bytes(b"mp4")

        with patch.object(substation, "resolve_binary", return_value="ffmpeg"), patch.object(
            substation, "run", side_effect=fake_run
        ):
            output = substation.embed(self.movie, self.srt, "es")

        self.assertEqual(output.read_bytes(), b"mp4")
        self.assertIn("language=spa", captured)

    def test_translation_preserves_timestamps_and_writes_separate_srt(self) -> None:
        self.srt.write_text(
            "1\n00:00:01,250 --> 00:00:02,500\nHello\n\n"
            "2\n00:00:04,000 --> 00:00:05,000\nAre you ready?\n",
            encoding="utf-8",
        )

        def fake_run(command, _step=None, **_kwargs) -> None:
            input_path = Path(command[command.index("--input") + 1])
            output_path = Path(command[command.index("--output") + 1])
            self.assertIn("Hello", input_path.read_text(encoding="utf-8"))
            output_path.write_text('["Hola", "¿Estás listo?"]', encoding="utf-8")

        with patch.object(substation, "check_translation_dependencies"), patch.object(
            substation, "_translation_worker_path", return_value=Path("worker.py")
        ), patch.object(substation, "run", side_effect=fake_run):
            output = substation.translate_srt(self.srt)

        translated = output.read_text(encoding="utf-8")
        self.assertEqual(output, self.temp_dir / "movie-es.srt")
        self.assertIn("00:00:01,250 --> 00:00:02,500", translated)
        self.assertIn("00:00:04,000 --> 00:00:05,000", translated)
        self.assertIn("¿Estás listo?", translated)

    def test_translation_rejects_wrong_cue_count_without_output(self) -> None:
        self.srt.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nWorld\n",
            encoding="utf-8",
        )

        def fake_run(command, _step=None, **_kwargs) -> None:
            output_path = Path(command[command.index("--output") + 1])
            output_path.write_text('["Hola"]', encoding="utf-8")

        with patch.object(substation, "check_translation_dependencies"), patch.object(
            substation, "_translation_worker_path", return_value=Path("worker.py")
        ), patch.object(substation, "run", side_effect=fake_run):
            with self.assertRaisesRegex(substation.EngineError, "different number of cues"):
                substation.translate_srt(self.srt)

        self.assertFalse((self.temp_dir / "movie-es.srt").exists())

    def test_translation_refuses_to_overwrite(self) -> None:
        self.srt.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nHello\n",
            encoding="utf-8",
        )
        output = self.temp_dir / "movie-es.srt"
        output.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(substation.EngineError, "not overwriting"):
            substation.translate_srt(self.srt)
        self.assertEqual(output.read_text(encoding="utf-8"), "keep")

    def test_exclusive_publish_preserves_destination(self) -> None:
        staged = self.temp_dir / "staged.srt"
        destination = self.temp_dir / "final.srt"
        staged.write_text("new", encoding="utf-8")
        destination.write_text("existing", encoding="utf-8")
        with self.assertRaisesRegex(substation.EngineError, "not overwriting"):
            substation._publish_new_file(staged, destination)
        self.assertEqual(destination.read_text(encoding="utf-8"), "existing")
        self.assertEqual(staged.read_text(encoding="utf-8"), "new")

    def test_parakeet_uses_explicit_srt_output_and_cue_limits(self) -> None:
        captured: list[str] = []

        def fake_run(command, _step=None, **_kwargs) -> None:
            captured.extend(command)
            # parakeet-mlx names its output after the input stem ("{filename}").
            output_dir = Path(command[command.index("--output-dir") + 1])
            (output_dir / f"{self.movie.stem}.srt").write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nHola\n",
                encoding="utf-8",
            )

        with patch.object(
            substation, "resolve_binary", return_value="parakeet-mlx"
        ) as resolver, patch.object(substation, "run", side_effect=fake_run):
            result = substation.transcribe(self.movie, "es", "parakeet")

        resolver.assert_called_once_with("parakeet-mlx")
        self.assertEqual(result, self.srt)
        self.assertEqual(captured[captured.index("--model") + 1], substation.PARAKEET_MODEL)
        self.assertEqual(captured[captured.index("--output-format") + 1], "srt")
        self.assertEqual(captured[captured.index("--output-template") + 1], "{filename}")
        self.assertEqual(captured[captured.index("--max-words") + 1], "14")
        self.assertNotIn("--language", captured)
        self.assertNotIn("--condition-on-previous-text", captured)
        self.assertFalse(any(path.name.startswith(".sub-station-") for path in self.temp_dir.iterdir()))

    def test_parakeet_success_exit_without_output_is_a_failure(self) -> None:
        # parakeet-mlx prints a per-file error and still exits 0.
        with patch.object(substation, "resolve_binary", return_value="parakeet-mlx"), patch.object(
            substation, "run", return_value=None
        ):
            with self.assertRaisesRegex(substation.EngineError, "was not created"):
                substation.transcribe(self.movie, engine="parakeet")

        self.assertFalse(self.srt.exists())
        self.assertFalse(any(path.name.startswith(".sub-station-") for path in self.temp_dir.iterdir()))

    def test_unknown_engine_is_rejected_before_any_output(self) -> None:
        with self.assertRaisesRegex(substation.EngineError, "unsupported speech engine"):
            substation.transcribe(self.movie, engine="vosk")
        self.assertEqual([path.name for path in self.temp_dir.iterdir()], ["movie.mkv"])

    def test_missing_parakeet_explains_one_time_setup(self) -> None:
        with patch.object(
            substation, "resolve_binary", side_effect=substation.EngineError("not found")
        ):
            with self.assertRaisesRegex(substation.EngineError, "parakeet-mlx>=0.4.1"):
                substation.check_parakeet_dependencies()

    def test_dedicated_whisper_binary_wins_over_inherited_path(self) -> None:
        dedicated = self.temp_dir / "mlx_whisper"
        dedicated.write_text("executable", encoding="utf-8")
        dedicated.chmod(0o700)
        with patch.dict(
            substation.DEDICATED_BINARIES,
            {"mlx_whisper": dedicated},
        ), patch.object(substation.shutil, "which", return_value="/wrong/mlx_whisper"):
            self.assertEqual(substation.resolve_binary("mlx_whisper"), str(dedicated))


class TranslationWorkerTests(unittest.TestCase):
    def test_multi_sentence_cue_is_split_without_breaking_titles(self) -> None:
        self.assertEqual(
            translation_worker.split_dialogue("Got one. Check this out."),
            ["Got one.", "Check this out."],
        )
        self.assertEqual(
            translation_worker.split_dialogue("Mr. Stone is here. Run!"),
            ["Mr. Stone is here.", "Run!"],
        )

    def test_spain_terms_are_normalized_to_neutral_latam(self) -> None:
        source = "Vale, vosotros tenéis el ordenador con zumos y patatas."
        self.assertEqual(
            translation_worker.normalize_latam(source),
            "Bien, ustedes tienen el computador con jugos y papas.",
        )

    def test_context_controls_mobile_car_and_stroller_terms(self) -> None:
        self.assertEqual(
            translation_worker.normalize_translation(
                "The mobile unit and mobile laboratory are ready.",
                "La unidad móvil y el laboratorio móvil están listos.",
            ),
            "La unidad móvil y el laboratorio móvil están listos.",
        )
        self.assertEqual(
            translation_worker.normalize_translation(
                "My phone is in the car with two phones.",
                "Mi móvil está en el coche con dos móviles.",
            ),
            "Mi celular está en el auto con dos celulares.",
        )
        self.assertEqual(
            translation_worker.normalize_translation(
                "Put it in the baby's stroller.",
                "Déjalo en el coche del bebé.",
            ),
            "Déjalo en el cochecito del bebé.",
        )

    def test_countdown_uses_neutral_latam_instruction(self) -> None:
        self.assertEqual(
            translation_worker.normalize_translation(
                "Okay, Tom and Maddie, count us down.",
                "Vale, Tom y Maddie, contadnos.",
            ),
            "Bien, Tom y Maddie, hagan la cuenta regresiva.",
        )

    def test_scientific_matter_uses_physics_terminology(self) -> None:
        self.assertEqual(
            translation_worker.normalize_translation(
                "The black hole will suck in all the matter.",
                "El agujero negro va a chupar todo el asunto.",
            ),
            "El agujero negro va a absorber toda la materia.",
        )

    def test_metal_failure_retries_complete_translation_on_cpu(self) -> None:
        fake_torch = types.SimpleNamespace(
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: True),
            ),
            mps=types.SimpleNamespace(empty_cache=lambda: None),
        )
        with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(
            translation_worker,
            "_translate_fragments",
            side_effect=[RuntimeError("simulated Metal failure"), ["Hola."]],
        ) as translator:
            self.assertEqual(translation_worker.translate(["Hello."], batch_size=1), ["Hola."])

        self.assertEqual(translator.call_args_list[0].kwargs["device"], "mps")
        self.assertEqual(translator.call_args_list[1].kwargs["device"], "cpu")


class GuiWorkflowTests(TemporaryDirectoryTest):
    def test_cancel_after_translation_recovers_spanish_output(self) -> None:
        movie = self.temp_dir / "movie.mp4"
        english = self.temp_dir / "movie.srt"
        spanish = self.temp_dir / "movie-es.srt"
        movie.write_bytes(b"movie")
        english.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nHello\n",
            encoding="utf-8",
        )
        cancel = threading.Event()
        fake_app = types.SimpleNamespace(
            _events=queue.Queue(),
            _queue_log=lambda _line: None,
        )

        def fake_translate(*_args, **_kwargs) -> Path:
            spanish.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nHola\n",
                encoding="utf-8",
            )
            cancel.set()
            return spanish

        with patch.object(substation_gui, "transcribe", return_value=english), patch.object(
            substation_gui, "clean_srt", return_value=(1, 0)
        ), patch.object(
            substation_gui, "find_repetition_warnings", return_value=[]
        ), patch.object(
            substation_gui, "translate_srt", side_effect=fake_translate
        ), patch.object(substation_gui, "embed") as embed_mock:
            substation_gui.SubStation._generate_worker(
                fake_app, movie, "en", "es", True, cancel
            )

        events = []
        while not fake_app._events.empty():
            events.append(fake_app._events.get())
        result = next(payload for event, payload in events if event == "done")
        self.assertEqual(result["state"], "cancelled")
        self.assertEqual(result["output"], spanish)
        self.assertIn(str(spanish), result["message"])
        embed_mock.assert_not_called()

    def test_cancel_during_embed_recovers_spanish_output(self) -> None:
        movie = self.temp_dir / "movie.mp4"
        english = self.temp_dir / "movie.srt"
        spanish = self.temp_dir / "movie-es.srt"
        movie.write_bytes(b"movie")
        english.write_text("English", encoding="utf-8")
        spanish.write_text("Spanish", encoding="utf-8")
        cancel = threading.Event()
        fake_app = types.SimpleNamespace(
            _events=queue.Queue(),
            _queue_log=lambda _line: None,
        )

        def fake_embed(*_args, **_kwargs) -> Path:
            cancel.set()
            raise substation.CancelledError("Embedding was cancelled.")

        with patch.object(substation_gui, "transcribe", return_value=english), patch.object(
            substation_gui, "clean_srt", return_value=(1, 0)
        ), patch.object(
            substation_gui, "find_repetition_warnings", return_value=[]
        ), patch.object(
            substation_gui, "translate_srt", return_value=spanish
        ), patch.object(substation_gui, "embed", side_effect=fake_embed):
            substation_gui.SubStation._generate_worker(
                fake_app, movie, "en", "es", True, cancel
            )

        events = []
        while not fake_app._events.empty():
            events.append(fake_app._events.get())
        result = next(payload for event, payload in events if event == "done")
        self.assertEqual(result["output"], spanish)
        self.assertIn(str(spanish), result["message"])

    def test_generate_worker_passes_selected_engine(self) -> None:
        movie = self.temp_dir / "movie.mp4"
        english = self.temp_dir / "movie.srt"
        movie.write_bytes(b"movie")
        english.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
        fake_app = types.SimpleNamespace(
            _events=queue.Queue(),
            _queue_log=lambda _line: None,
        )

        with patch.object(
            substation_gui, "transcribe", return_value=english
        ) as transcribe_mock, patch.object(
            substation_gui, "clean_srt", return_value=(1, 0)
        ), patch.object(substation_gui, "find_repetition_warnings", return_value=[]):
            substation_gui.SubStation._generate_worker(
                fake_app, movie, "en", "en", False, threading.Event(), "parakeet"
            )

        self.assertEqual(transcribe_mock.call_args.args, (movie, "en", "parakeet"))
        events = []
        while not fake_app._events.empty():
            events.append(fake_app._events.get())
        self.assertIn(("stage", "Transcribing audio with Parakeet…"), events)
        result = next(payload for event, payload in events if event == "done")
        self.assertEqual(result["state"], "success")

    def test_unready_parakeet_blocks_job_and_shows_setup(self) -> None:
        begin_job = Mock()
        fake_app = types.SimpleNamespace(
            _valid_movie=lambda: self.temp_dir / "movie.mp4",
            _selected_mode=lambda: ("en", "en"),
            _selected_engine=lambda: "parakeet",
            _translator_ready=False,
            _tools_ready=True,
            _parakeet_ready=False,
            _parakeet_detail="Parakeet is not installed. Set it up once.",
            _begin_job=begin_job,
        )

        with patch.object(substation_gui.messagebox, "showerror") as error:
            substation_gui.SubStation._start_generate(fake_app)

        self.assertIn("Set it up once", error.call_args.args[1])
        begin_job.assert_not_called()

    def test_clean_gui_reports_each_removed_category(self) -> None:
        source = self.temp_dir / "source.srt"
        output = self.temp_dir / "source-clean.srt"
        source.write_text("source", encoding="utf-8")
        output.write_text("clean", encoding="utf-8")
        status: dict[str, str] = {}
        fake_widget = types.SimpleNamespace(configure=lambda **kwargs: None)
        fake_app = types.SimpleNamespace(
            srt_var=types.SimpleNamespace(get=lambda: str(source)),
            _last_output=None,
            clean_reveal_button=fake_widget,
            new_job_button=fake_widget,
            clean_status=types.SimpleNamespace(configure=lambda **kwargs: status.update(kwargs)),
        )

        with patch.object(
            substation_gui, "clean_ads", return_value=(output, 1, 0, 1)
        ), patch.object(substation_gui.messagebox, "showinfo") as info:
            substation_gui.SubStation._clean_srt(fake_app)

        self.assertIn("0 promo lines", status["text"])
        self.assertIn("1 invalid/empty cues", status["text"])
        self.assertIn("0 promotional lines", info.call_args.args[1])
        self.assertIn("1 invalid or empty cues", info.call_args.args[1])


class ProcessRunnerTests(unittest.TestCase):
    def test_failure_contains_tool_output(self) -> None:
        with self.assertRaisesRegex(substation.EngineError, "specific failure"):
            substation.run(
                [sys.executable, "-c", "print('specific failure'); raise SystemExit(3)"],
                "Test tool",
            )

    def test_cancel_stops_process_group(self) -> None:
        cancel = threading.Event()
        timer = threading.Timer(0.2, cancel.set)
        timer.start()
        try:
            with self.assertRaises(substation.CancelledError):
                substation.run(
                    [sys.executable, "-c", "import time; print('start', flush=True); time.sleep(30)"],
                    "Long tool",
                    cancel_event=cancel,
                )
        finally:
            timer.cancel()


if __name__ == "__main__":
    unittest.main()
