from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'small_bench').resolve()
    errors: list[str] = []

    manifest_path = root / 'manifest.jsonl'
    refs_path = root / 'references.jsonl'
    identity_path = root / 'dataset_identity.json'

    for path in [manifest_path, refs_path, identity_path, root / 'typed.json', root / 'hand-written.json']:
        if not path.exists():
            errors.append(f'missing: {path}')

    if errors:
        print('\n'.join(errors))
        return 1

    manifest = [json.loads(x) for x in manifest_path.read_text(encoding='utf-8').splitlines() if x.strip()]
    refs = [json.loads(x) for x in refs_path.read_text(encoding='utf-8').splitlines() if x.strip()]
    ref_by_id = {x['sample_id']: x for x in refs}

    if len(manifest) != 20:
        errors.append(f'expected 20 manifest rows, got {len(manifest)}')
    if len(refs) != 20:
        errors.append(f'expected 20 reference rows, got {len(refs)}')
    if len(ref_by_id) != len(refs):
        errors.append('duplicate sample_id in references')

    dataset_hashes = {x['dataset_sha256'] for x in manifest}
    if len(dataset_hashes) != 1:
        errors.append('manifest rows do not share one dataset_sha256')

    identity_material = {
        'dataset_id': manifest[0]['dataset_id'] if manifest else None,
        'dataset_version': manifest[0]['dataset_version'] if manifest else None,
        'samples': [],
    }

    for row in manifest:
        sid = row['sample_id']
        image = root / row['image']
        if not image.exists():
            errors.append(f'{sid}: missing image {image}')
            continue
        if sha256(image) != row['image_sha256']:
            errors.append(f'{sid}: image hash mismatch')
        try:
            with Image.open(image) as im:
                im.verify()
        except Exception as exc:
            errors.append(f'{sid}: unreadable image: {exc}')

        ref = ref_by_id.get(sid)
        if not ref:
            errors.append(f'{sid}: missing reference')
            continue
        scorable_hash = hashlib.sha256(ref['scorable_text'].encode('utf-8')).hexdigest()
        norm_hash = hashlib.sha256(ref['fa_ir_normalized_text'].encode('utf-8')).hexdigest()
        if scorable_hash != row['reference_scorable_sha256'] or scorable_hash != ref['scorable_text_sha256']:
            errors.append(f'{sid}: scorable reference hash mismatch')
        if norm_hash != row['reference_normalized_sha256'] or norm_hash != ref['fa_ir_normalized_sha256']:
            errors.append(f'{sid}: normalized reference hash mismatch')

        identity_material['samples'].append({
            'sample_id': sid,
            'image_sha256': row['image_sha256'],
            'reference_scorable_sha256': row['reference_scorable_sha256'],
            'metadata_sha256': row['metadata_sha256'],
        })

    calculated = hashlib.sha256(stable_json(identity_material).encode('utf-8')).hexdigest()
    identity = json.loads(identity_path.read_text(encoding='utf-8'))
    if calculated != identity.get('dataset_sha256'):
        errors.append('dataset identity hash mismatch')
    if dataset_hashes and calculated != next(iter(dataset_hashes)):
        errors.append('manifest dataset_sha256 mismatch')

    compatibility_count = 0
    for filename in ['typed.json', 'hand-written.json']:
        data = json.loads((root / filename).read_text(encoding='utf-8'))
        compatibility_count += len(data)
        for item in data:
            if not (root / item['image']).exists():
                errors.append(f'{filename}: missing image {item["image"]}')
    if compatibility_count != 20:
        errors.append(f'compatibility JSON count is {compatibility_count}, expected 20')

    if errors:
        print('FAILED')
        for error in errors:
            print(f'- {error}')
        return 1

    print('OK')
    print(f'dataset_id={identity["dataset_id"]}')
    print(f'dataset_version={identity["dataset_version"]}')
    print(f'dataset_sha256={identity["dataset_sha256"]}')
    print(f'samples={identity["sample_count"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
