# Data and checkpoints

Large external inputs are deliberately excluded from this repository. Keep them in
a separate pinned TRM checkout and expose that path with `TRM_ROOT`.

## Upstream code

```bash
git clone https://github.com/AntonioRoye/TinyRecursiveModels.git TinyRecursiveModels
git -C TinyRecursiveModels checkout 010206d1f0c25ebac0865f69e39c09969e6b896b
export TRM_ROOT="$(cd TinyRecursiveModels && pwd)"
```

The validated TRM commit is
`010206d1f0c25ebac0865f69e39c09969e6b896b`. Do not substitute a moving branch.
The ARC-AGI-2 reference clone used during provenance review was at
`f3283f727488ad98fe575ea6a5ac981e4a188e49`.

## Checkpoints

Download `arcprize/trm_arc_prize_verification` from Hugging Face at exact revision
`55ced5dd59de74c52f53d47aa2898232b5a15b7a`:

```bash
python -m pip install 'huggingface_hub[cli]'
hf download arcprize/trm_arc_prize_verification \
  --revision 55ced5dd59de74c52f53d47aa2898232b5a15b7a \
  --local-dir "$TRM_ROOT/checkpoints/hf_trm"
```

| Checkpoint | Expected bytes | SHA256 |
|---|---:|---|
| `arc_v1_public/step_518071` | 1,822,205,258 | `53689643ad1606d7c22c758f8af0a71b3b66275dea074f214d2f1048d9a01fb0` |
| `arc_v2_public/step_723914` | 2,467,988,810 | `8d7036b97e7ea38c7dd29d01216bfcfc4e212af3024d5233fe40dd3059e8f4a9` |

Verify after download:

```bash
sha256sum \
  "$TRM_ROOT/checkpoints/hf_trm/arc_v1_public/step_518071" \
  "$TRM_ROOT/checkpoints/hf_trm/arc_v2_public/step_723914"
```

## Raw and processed ARC data

Use the pinned checkout's `kaggle/combined/arc-agi` inventory. File ordering is part
of puzzle-embedding alignment; do not replace individual JSON files from another
checkout without rebuilding and rerunning the identifier gate.

Build the two datasets separately:

```bash
TRM_ROOT="$TRM_ROOT" python experiments/prepare_arc_datasets.py \
  --dataset arc1 --seed 42 --num-aug 1000

TRM_ROOT="$TRM_ROOT" python experiments/prepare_arc_datasets.py \
  --dataset arc2 --seed 42 --num-aug 1000
```

Strict recipes:

| Dataset | Subsets | Test set | Identifiers | Observed directory bytes | `identifiers.json` SHA256 |
|---|---|---|---:|---:|---|
| ARC1 | `training evaluation concept` | `evaluation` | 876,406 | 7,273,795,728 | `9f42efad9afc8796ba45dafa973e19484bc7d9006a847eba632453ff397b1ac9` |
| ARC2 | `training2 evaluation2 concept` | `evaluation2` | 1,191,730 | 9,648,612,450 | `d7ff9bada1e1b449013a475ea6cd83d4dfda6c38ad40ca58940c2c7e6ad3aca3` |

The preparation script refuses to overwrite a non-empty output directory and writes
a build manifest containing the recipe and pinned code revision.

## Prediction caches

Prediction caches are also excluded, even though the compressed NPZ inventories are
relatively compact. Expected paths in a reproduction workspace are:

```text
artifacts/cache/arc1_normal
artifacts/cache/arc1_blank
artifacts/cache/arc2_normal
```

| Cache | NPZ files | NPZ bytes | Aggregate fingerprint |
|---|---:|---:|---|
| ARC1 normal | 419 | 14,655,304 | `6cb320cb9b0e45c114d95722654663d2b7e98b01027a1d4a647c21ea4ce60fa1` |
| ARC1 blank | 419 | 17,122,246 | `9982f59fd5b952f6e80430adadebf047b107d357a18c04abf3cc30b204f0e5bd` |
| ARC2 normal | 172 | 9,825,347 | `81f5f077adcc35770053c162486ea98c19953b1a94912f4dccf3998021043d20` |

The aggregate fingerprint hashes each sorted filename and its file SHA256, matching
`orbit_consensus.supporting_analysis._cache_fingerprint`.

External repositories, datasets, and checkpoints retain their original licenses.
This release does not redistribute them.
