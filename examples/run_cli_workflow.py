#!/usr/bin/env python3
"""Run a complete BellLab CLI workflow on temporary synthetic WAV files.

The script demonstrates the public command-line interface without relying on
personal paths, network access, notebooks or an interactive display. Scientific
analysis is run once by ``belllab analyze``; export, visualization and report
commands reuse the saved analysis bundle.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="belllab-cli-workflow-") as temp_dir:
        root = Path(temp_dir)
        audio_dir = root / "audio"
        _write_wavs(audio_dir)

        analysis = root / "analysis.json"
        export_bundle = root / "export-bundle.json"
        figures_bundle = root / "figures-bundle.json"
        report_bundle = root / "report-bundle.json"

        commands = [
            [
                "analyze",
                "--recording",
                f"pp={audio_dir / 'pp.wav'}",
                "--recording",
                f"p={audio_dir / 'p.wav'}",
                "--recording",
                f"mf={audio_dir / 'mf.wav'}",
                "--recording",
                f"f={audio_dir / 'f.wav'}",
                "--recording",
                f"ff={audio_dir / 'ff.wav'}",
                "--until-stage",
                "global_spectrum",
                "--save-result",
                str(analysis),
                "--output-format",
                "json",
            ],
            [
                "export",
                "--analysis",
                str(analysis),
                "--output-dir",
                str(root / "export"),
                "--json",
                "--csv",
                "--manifest",
                "--overwrite",
                "replace",
                "--save-result",
                str(export_bundle),
                "--output-format",
                "json",
            ],
            [
                "visualize",
                "--analysis",
                str(analysis),
                "--output-dir",
                str(root / "figures"),
                "--figure",
                "global_spectrum",
                "--format",
                "png",
                "--format",
                "svg",
                "--overwrite",
                "replace",
                "--save-result",
                str(figures_bundle),
                "--output-format",
                "json",
            ],
            [
                "report",
                "--analysis",
                str(analysis),
                "--export-result",
                str(export_bundle),
                "--figure-collection",
                str(figures_bundle),
                "--output-dir",
                str(root / "report"),
                "--markdown",
                "--latex",
                "--overwrite",
                "replace",
                "--title",
                "BellLab CLI workflow report",
                "--author",
                "BellLab example",
                "--save-result",
                str(report_bundle),
                "--output-format",
                "json",
            ],
            [
                "inspect",
                str(report_bundle),
                "--validate",
                "--output-format",
                "json",
            ],
        ]

        summaries = []
        for command in commands:
            completed = _run(command)
            payload = json.loads(completed.stdout)
            summaries.append({
                "command": command[0],
                "exit_code": completed.returncode,
                "status": payload["status"],
                "artifact_count": len(payload.get("artifact_paths", ())),
            })
        print(json.dumps(summaries, indent=2, sort_keys=True))
        return 0


def _run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "belllab", *arguments]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode not in {0, 1, 5}:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(completed.returncode)
    return completed


def _write_wavs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    sample_rate = 4096
    duration_s = 1.0
    time = np.arange(int(sample_rate * duration_s), dtype=np.float64) / sample_rate
    for index, label in enumerate(("pp", "p", "mf", "f", "ff")):
        frequency = 300.0 + index
        signal = 0.3 * np.exp(-time / 2.0) * np.sin(2 * np.pi * frequency * time)
        sf.write(directory / f"{label}.wav", signal.astype(np.float32), sample_rate, subtype="FLOAT")


if __name__ == "__main__":
    raise SystemExit(main())
