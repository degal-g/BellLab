"""Check the basic metadata of a built BellLab wheel using only the standard library."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from pathlib import Path
import re
import zipfile


def _pyproject_value(name: str) -> str:
    pattern = re.compile(rf'^\s*{re.escape(name)}\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
    match = pattern.search(Path("pyproject.toml").read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"could not read {name!r} from pyproject.toml")
    return match.group(1)


def check_wheel(wheel: Path) -> None:
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise ValueError(f"wheel not found: {wheel}")
    with zipfile.ZipFile(wheel) as archive:
        metadata_files = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise ValueError(f"expected one METADATA file in {wheel}, found {len(metadata_files)}")
        metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
    expected = {"Name": _pyproject_value("name"), "Version": _pyproject_value("version"), "Requires-Python": _pyproject_value("requires-python")}
    for field, value in expected.items():
        actual = metadata.get(field)
        if actual != value:
            raise ValueError(f"{field} mismatch: expected {value!r}, got {actual!r}")
    print(f"metadata OK: {wheel.name} ({expected['Name']} {expected['Version']}, Python {expected['Requires-Python']})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    try:
        check_wheel(args.wheel)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
