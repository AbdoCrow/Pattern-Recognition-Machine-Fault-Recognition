r"""
EDA script for machine audio classification.

Usage example:
    python eda_audio.py --src "c:\pattern\data\categorized" --samples-per-class 3

This script requires `librosa` and `numpy`.
If not installed: pip install librosa numpy soundfile
"""
import argparse
from pathlib import Path
import random
import numpy as np


def try_imports():
    try:
        import librosa
    except Exception as e:
        print("ERROR: librosa is required. Install with: pip install librosa soundfile")
        raise
    return librosa


def rms(y):
    return float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0


def analyze_file(librosa, filepath: Path, top_db: int, noise_seconds: float):
    y, sr = librosa.load(str(filepath), sr=None, mono=True)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    overall_rms = rms(y)
    raw_duration = len(y) / sr if sr else 0.0

    # trim leading/trailing silence
    trimmed, intervals = librosa.effects.trim(y, top_db=top_db)
    trimmed_duration = len(trimmed) / sr if sr else 0.0
    silence_removed = max(0.0, raw_duration - trimmed_duration)

    # noise floor estimate from first N seconds
    n_noise = int(noise_seconds * sr)
    n_noise = min(n_noise, len(y))
    noise_rms = rms(y[:n_noise]) if n_noise > 0 else 0.0

    return {
        'filepath': str(filepath),
        'sr': int(sr),
        'raw_duration_s': float(raw_duration),
        'trimmed_duration_s': float(trimmed_duration),
        'silence_removed_s': float(silence_removed),
        'peak': float(peak),
        'rms': float(overall_rms),
        'noise_rms_{}s'.format(noise_seconds): float(noise_rms),
    }


def print_metrics(class_name: str, results: list):
    print('\n=== Class: {} ==='.format(class_name))
    print('Samples analyzed:', len(results))
    if not results:
        print('  (no .wav files found)')
        return
    srs = sorted({r['sr'] for r in results})
    print('Sampling rates found:', srs)

    for i, r in enumerate(results, 1):
        print('\n  Sample {}: {}'.format(i, r['filepath']))
        print('    - sr: {} Hz'.format(r['sr']))
        print('    - raw duration: {:.3f} s'.format(r['raw_duration_s']))
        print('    - trimmed duration (top_db trimming): {:.3f} s'.format(r['trimmed_duration_s']))
        print('    - silence removed: {:.3f} s'.format(r['silence_removed_s']))
        print('    - peak amplitude: {:.6f}'.format(r['peak']))
        print('    - RMS amplitude: {:.6f}'.format(r['rms']))
        # find noise key
        noise_key = [k for k in r.keys() if k.startswith('noise_rms_')][0]
        print('    - noise RMS (first segment): {:.6f}'.format(r[noise_key]))


def main():
    librosa = try_imports()

    parser = argparse.ArgumentParser(description='EDA for machine audio dataset')
    parser.add_argument('--src', required=True, help='Root folder with 6 class subfolders')
    parser.add_argument('--samples-per-class', type=int, default=3, help='How many random wavs per class to analyze')
    parser.add_argument('--top-db', type=int, default=20, help='top_db for librosa.effects.trim')
    parser.add_argument('--topdbs', type=str, default='', help='Comma-separated list of top_db values to test (e.g. 20,30,40)')
    parser.add_argument('--noise-seconds', type=float, default=0.5, help='Seconds from start to estimate noise RMS')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for sampling')
    args = parser.parse_args()

    random.seed(args.seed)
    src = Path(args.src)
    if not src.exists():
        print('Source folder not found:', src)
        return

    # Gather class folders (only immediate children)
    class_folders = [p for p in sorted(src.iterdir()) if p.is_dir()]
    if not class_folders:
        print('No class subfolders found under', src)
        return

    # Class distribution
    print('\nClass distribution:')
    class_counts = {}
    for cf in class_folders:
        wavs = list(cf.rglob('*.wav'))
        class_counts[cf.name] = len(wavs)
        print('  - {} : {} wav files'.format(cf.name, len(wavs)))

    # Analyze a few random samples per class
    for cf in class_folders:
        wav_files = list(cf.rglob('*.wav'))
        if not wav_files:
            print('\nSkipping {} (no wavs)'.format(cf.name))
            continue
        k = min(args.samples_per_class, len(wav_files))
        chosen = random.sample(wav_files, k)
        results = []
        for w in chosen:
            try:
                if args.topdbs:
                    topdb_list = [int(x) for x in args.topdbs.split(',') if x.strip()]
                else:
                    topdb_list = [args.top_db]

                multi_results = []
                for td in topdb_list:
                    r = analyze_file(librosa, w, top_db=td, noise_seconds=args.noise_seconds)
                    r['_top_db'] = td
                    multi_results.append(r)
                results.append(multi_results)
            except Exception as e:
                print('Error analyzing', w, '-', e)

        # Print results: if we tested multiple top_db values, results is list of lists
        print('\n=== Class: {} ==='.format(cf.name))
        print('Samples analyzed:', len(results))
        if not results:
            print('  (no .wav files found)')
            continue

        # show sampling rates
        srs = sorted({mr[0]['sr'] for mr in results})
        print('Sampling rates found:', srs)

        for i, mr_list in enumerate(results, 1):
            base = mr_list[0]
            print('\n  Sample {}: {}'.format(i, base['filepath']))
            print('    - sr: {} Hz'.format(base['sr']))
            print('    - peak amplitude: {:.6f}'.format(base['peak']))
            print('    - RMS amplitude: {:.6f}'.format(base['rms']))
            noise_key = [k for k in base.keys() if k.startswith('noise_rms_')][0]
            print('    - noise RMS (first segment): {:.6f}'.format(base[noise_key]))
            print('    - trimmed results for top_db values:')
            for r in mr_list:
                print('       * top_db={}: raw {:.3f}s -> trimmed {:.3f}s (silence removed {:.3f}s)'.format(
                    r['_top_db'], r['raw_duration_s'], r['trimmed_duration_s'], r['silence_removed_s']))

        # ===== Compute averages per top_db =====
        from collections import defaultdict

        avg_stats = defaultdict(lambda: {
            'raw': [],
            'trimmed': [],
            'silence': []
        })

        for mr_list in results:
            for r in mr_list:
                td = r['_top_db']
                avg_stats[td]['raw'].append(r['raw_duration_s'])
                avg_stats[td]['trimmed'].append(r['trimmed_duration_s'])
                avg_stats[td]['silence'].append(r['silence_removed_s'])

        print("\nAverages per top_db:")
        for td, vals in sorted(avg_stats.items()):
            raw_avg = sum(vals['raw']) / len(vals['raw'])
            trimmed_avg = sum(vals['trimmed']) / len(vals['trimmed'])
            silence_avg = sum(vals['silence']) / len(vals['silence'])

            print(f"   🔹 top_db={td}:")
            print(f"      avg raw duration     = {raw_avg:.3f}s")
            print(f"      avg trimmed duration = {trimmed_avg:.3f}s")
            print(f"      avg silence removed  = {silence_avg:.3f}s")
if __name__ == '__main__':
    main()
