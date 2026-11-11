"""Build a fixed manifest of 600 original RGB images from the training ZIP directory."""
import json
import random
from pathlib import Path

def main():
    rows = json.loads(Path('.work/train-central.json').read_text())
    ordered = sorted(rows, key=lambda r: r['offset'])
    for i, row in enumerate(ordered):
        row['span_end'] = (ordered[i + 1]['offset'] if i + 1 < len(ordered) else 1691153006) - 1
    categories = ['Negative', 'Dry_arable', 'Continuos_discontinuos_urban',
                  'Desert_bare', 'Agricoltural_areas', 'Water']
    rng = random.Random(42)
    selected = []
    batch = 0
    for category in categories:
        candidates = [r for r in ordered if Path(r['member']).parent.name == category]
        blocks = [candidates[i:i + 10] for i in range(0, len(candidates) - 9, 10)]
        chosen = rng.sample(blocks, 30 if category == 'Negative' else 6)
        for block in chosen:
            for row in block:
                selected.append(dict(row, batch=batch))
            batch += 1
    manifest = {'source': 'https://zenodo.org/records/14950565',
                'archive': 'Normalized_Tell_combined_350.zip',
                'published_archive_md5': '1d4ea398ea5669f9bca4539416847206',
                'sampling': '600 images: 300 negatives; 60 from each of five unmasked RGB positive context folders. Seed 42; blocks of ten for efficient HTTP ranges.',
                'excluded_positive_folders': ['P_BW', 'Masked'],
                'members': selected}
    Path('.work/sample_manifest.json').write_text(json.dumps(manifest, indent=2))
    size = sum(max(r['span_end'] for r in selected if r['batch'] == b) -
               min(r['offset'] for r in selected if r['batch'] == b) + 1 for b in range(batch))
    print(f'{len(selected)} images, {batch} range requests, {size / 1e6:.1f} MB transfer')


if __name__ == "__main__":
    main()
