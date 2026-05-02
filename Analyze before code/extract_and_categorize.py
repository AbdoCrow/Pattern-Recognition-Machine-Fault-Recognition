import argparse
import zipfile
import shutil
import tempfile
from pathlib import Path
import re


def detect_machine(name: str):
    m = re.search(r"Machine\s*([0-9])", name, re.IGNORECASE)
    return m.group(1) if m else None


def detect_state(name: str):
    name_low = name.lower()
    if "abnormal" in name_low or "fault" in name_low:
        return "abnormal"
    if "normal" in name_low or "healthy" in name_low:
        return "normal"
    return None


def process_zip(zippath: Path, dst_root: Path, copy_files: bool = True):
    machine = detect_machine(zippath.name) or "unknown"
    with zipfile.ZipFile(zippath, 'r') as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            member_path = Path(info.filename)
            filename = member_path.name
            parent_name = member_path.parent.name
            state = detect_state(filename) or detect_state(parent_name) or "unknown"
            category = f"Machine{machine}_{state}"
            dest_dir = dst_root / category
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / filename
            with z.open(info) as src, open(dest_file, 'wb') as out:
                shutil.copyfileobj(src, out)


def main():
    parser = argparse.ArgumentParser(description="Extract zip files and categorize by machine/state")
    parser.add_argument('--src', required=True, help='Source folder containing zip files')
    parser.add_argument('--dst', required=True, help='Destination root for categorized folders')
    parser.add_argument('--move', action='store_true', help='Move files instead of copying')
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    zips = sorted(src.glob('*.zip'))
    if not zips:
        print('No zip files found in', src)
        return

    for z in zips:
        print('Processing', z.name)
        process_zip(z, dst, copy_files=not args.move)

    print('Done. Categorized files are under', dst)


if __name__ == '__main__':
    main()
