#!/usr/bin/env python
"""Convert legacy full-pickle checkpoints (with custom classes) into a safe, portable
weights-only style checkpoint usable with torch.load(..., weights_only=True).

Usage:
  python convert_checkpoint.py --input experiments/uspto_50k/without_rxn_class/12-05-2024--09-48-03/epoch_120.pt \
      --output experiments/uspto_50k/without_rxn_class/12-05-2024--09-48-03/epoch_120_safe.pt --verify

If --output not provided, will append _safe before extension.

The script expects the original checkpoint dict to contain at least keys:
  'saveables' -> model config kwargs
  'state'     -> state_dict tensor mapping (or something loadable into model)

It will re-instantiate Graph2Edits with the config, load weights, then resave a minimal dict:
  {
     'format_version': 1,
     'saveables': <config>,
     'state': <state_dict>
  }

This avoids pickling arbitrary Python objects (e.g., custom vocab classes) and
lets you later load with weights_only=True safely.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import torch
import subprocess
import concurrent.futures as cf

# Local imports
from models import Graph2Edits
from utils.rxn_graphs import Vocab  # for type identification


def parse_args():
    p = argparse.ArgumentParser(description="Convert legacy checkpoint(s) to safe format_version=2 weights-only style.")
    # Single-file mode
    p.add_argument('--input', '-i', help='Path to legacy checkpoint .pt/.pth (single file mode)')
    p.add_argument('--output', '-o', required=False, help='Output path for safe checkpoint (single file mode)')
    # Batch mode
    p.add_argument('--batch-root', help='Root directory to recursively scan for legacy checkpoints')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing *_safe.pt if already exists (batch)')
    p.add_argument('--workers', type=int, default=1, help='Parallel workers for batch conversion')
    p.add_argument('--dry-run', action='store_true', help='List planned conversions without executing (batch)')
    # Shared
    p.add_argument('--device', default='cpu', help='Device to instantiate model (cpu or cuda)')
    p.add_argument('--verify', action='store_true', help='After saving, attempt torch.load(weights_only=True) to verify')
    return p.parse_args()


def infer_output_path(inp: Path) -> Path:
    if inp.suffix:
        return inp.with_name(inp.stem + '_safe' + inp.suffix)
    return inp.with_name(inp.name + '_safe.pt')


def convert_single(in_path: Path, out_path: Path, verify: bool):
    if not in_path.is_file():
        raise FileNotFoundError(f'Input checkpoint not found: {in_path}')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'[INFO] Loading legacy checkpoint: {in_path}')
    legacy_ckpt = torch.load(in_path, map_location='cpu', weights_only=False)
    if 'saveables' not in legacy_ckpt or 'state' not in legacy_ckpt:
        raise KeyError('Checkpoint missing required keys: saveables / state')
    config = legacy_ckpt['saveables']
    state = legacy_ckpt['state']
    if isinstance(config, dict) and 'config' in config and 'atom_vocab' in config and 'bond_vocab' in config:
        inner_cfg = config['config']
        atom_vocab_obj = config['atom_vocab']
        bond_vocab_obj = config['bond_vocab']
    else:
        inner_cfg = config.get('config', None)
        atom_vocab_obj = config.get('atom_vocab', None)
        bond_vocab_obj = config.get('bond_vocab', None)
        if inner_cfg is None:
            temp_cfg = {k: v for k, v in config.items() if k not in ['atom_vocab', 'bond_vocab']}
            if 'n_atom_feat' in temp_cfg or 'mpn_size' in temp_cfg:
                inner_cfg = temp_cfg
    if inner_cfg is None or atom_vocab_obj is None or bond_vocab_obj is None:
        raise KeyError('Unable to parse config / atom_vocab / bond_vocab from checkpoint; inspect original saveables structure.')
    if not isinstance(atom_vocab_obj, Vocab) or not isinstance(bond_vocab_obj, Vocab):
        raise TypeError('atom_vocab or bond_vocab is not a Vocab instance; cannot safely extract.')
    atom_vocab_list = list(atom_vocab_obj.elem_list)
    bond_vocab_list = list(bond_vocab_obj.elem_list)
    print('[INFO] Instantiating model to validate state dict...')
    model = Graph2Edits(config=inner_cfg, atom_vocab=atom_vocab_obj, bond_vocab=bond_vocab_obj, device='cpu')
    model.load_state_dict(state)
    safe_ckpt = {
        'format_version': 2,
        'model_args': {
            'config': inner_cfg,
            'atom_vocab_list': atom_vocab_list,
            'bond_vocab_list': bond_vocab_list,
        },
        'state': model.state_dict(),
    }
    torch.save(safe_ckpt, out_path)
    print(f'[OK] Saved safe checkpoint (format_version=2): {out_path}')
    if verify:
        print('[INFO] Verifying weights_only load (no custom class should be required)...')
        test = torch.load(out_path, map_location='cpu', weights_only=True)
        for k in ['format_version', 'model_args', 'state']:
            if k not in test:
                raise AssertionError(f'Verification failed, key missing: {k}')
        if test['format_version'] != 2:
            raise AssertionError('Verification failed: format_version mismatch.')
        ma = test['model_args']
        for sub in ['config', 'atom_vocab_list', 'bond_vocab_list']:
            if sub not in ma:
                raise AssertionError(f'Verification failed: model_args missing {sub}')
        print('[OK] Verification succeeded for format_version=2.')


def is_candidate(path: Path) -> bool:
    if not path.name.endswith('.pt'):
        return False
    if path.name.endswith('_safe.pt'):
        return False
    return True


def batch_convert(root: Path, overwrite: bool, workers: int, dry_run: bool, verify: bool):
    if not root.exists():
        print(f'[ERR] root not found: {root}')
        sys.exit(1)
    candidates = [p for p in root.rglob('*.pt') if is_candidate(p)]
    if not candidates:
        print('[INFO] No legacy candidates found.')
        return
    print(f'[INFO] Found {len(candidates)} candidate checkpoint(s).')
    for p in candidates:
        print('  -', p)
    if dry_run:
        print('[DRY-RUN] Nothing converted.')
        return
    def _do(p: Path):
        outp = p.with_name(p.stem + '_safe.pt')
        if outp.exists() and not overwrite:
            return p, outp, True, 'exists (skipped)'
        try:
            convert_single(p, outp, verify)
            return p, outp, True, 'ok'
        except Exception as e:  # pragma: no cover
            return p, outp, False, str(e)[:400]
    if workers > 1:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_do, p) for p in candidates]
            for fut in cf.as_completed(futures):
                inp, outp, ok, msg = fut.result()
                status = 'OK' if ok else 'FAIL'
                print(f'[{status}] {inp.name} -> {outp.name}: {msg}')
    else:
        for p in candidates:
            inp, outp, ok, msg = _do(p)
            status = 'OK' if ok else 'FAIL'
            print(f'[{status}] {inp.name} -> {outp.name}: {msg}')
    print('[DONE] Batch conversion finished.')


def main():
    args = parse_args()
    # Determine mode
    if args.batch_root:
        batch_convert(Path(args.batch_root), args.overwrite, args.workers, args.dry_run, args.verify)
        return
    if not args.input:
        print('Either provide --input for single-file mode or --batch-root for batch mode.')
        sys.exit(2)
    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else infer_output_path(in_path)
    convert_single(in_path, out_path, args.verify)
    print('[DONE] Conversion complete.')


if __name__ == '__main__':
    main()
