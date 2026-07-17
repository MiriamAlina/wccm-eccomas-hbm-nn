"""Interactive selection helpers for generated data and trained models."""

from __future__ import annotations

import os
import re
import select
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


_TIMESTAMP_RE = re.compile(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}')


@dataclass(frozen=True)
class ArtifactCandidate:
    """A selectable file-backed artifact."""

    artifact_id: str
    path: Path
    timestamp: datetime | None

    @property
    def modified_at(self) -> datetime:
        return datetime.fromtimestamp(self.path.stat().st_mtime)


def select_data_id(
    data_dir: str | Path = 'data',
    *,
    h: int = 3,
    require_splits: tuple[str, ...] = ('train', 'val', 'test'),
    timeout_seconds: int = 15,
) -> str:
    """Select a generated data-set id from ``data_dir``.

    The selection is based on files named ``<split>_data_H<h>_<id>.npz``.
    By default, only ids with train/val/test files are offered because model
    training requires all three splits.
    """
    label = f'data set H{h}'
    data_dir = Path(data_dir)
    pattern = f'train_data_H{h}_*.npz'

    def is_complete(candidate: ArtifactCandidate) -> bool:
        return all(
            (data_dir / f'{split}_data_H{h}_{candidate.artifact_id}.npz')
            .is_file()
            for split in require_splits
        )

    return _select_artifact_id(
        directory=data_dir,
        pattern=pattern,
        id_prefix=f'train_data_H{h}_',
        id_suffix='.npz',
        label=label,
        timeout_seconds=timeout_seconds,
        filter_candidate=is_complete,
    )


def select_model_id(
    models_dir: str | Path = 'models',
    *,
    model_stem: str = 'jenkins_h3',
    timeout_seconds: int = 15,
) -> str:
    """Select a trained model id from ``models_dir``.

    The selection is based on ``mlp_<model_stem>_<id>.pt`` and only offers
    models with matching scaling parameters, because inference needs both.
    """
    label = model_stem.replace('_', ' ')
    models_dir = Path(models_dir)
    pattern = f'mlp_{model_stem}_*.pt'

    def has_scaler(candidate: ArtifactCandidate) -> bool:
        return (
            models_dir
            / f'scaling_params_{model_stem}_{candidate.artifact_id}.joblib'
        ).is_file()

    return _select_artifact_id(
        directory=models_dir,
        pattern=pattern,
        id_prefix=f'mlp_{model_stem}_',
        id_suffix='.pt',
        label=f'model {label}',
        timeout_seconds=timeout_seconds,
        filter_candidate=has_scaler,
    )


def _select_artifact_id(
    *,
    directory: Path,
    pattern: str,
    id_prefix: str,
    id_suffix: str,
    label: str,
    timeout_seconds: int,
    filter_candidate,
) -> str:
    candidates = [
        candidate
        for candidate in _find_candidates(directory, pattern, id_prefix,
                                          id_suffix)
        if filter_candidate(candidate)
    ]
    if not candidates:
        raise FileNotFoundError(
            f'No matching {label} found in {directory} for pattern {pattern}.'
        )

    # Newest timestamp in the file name wins. The modification time is a
    # fallback/tie-breaker for files without a parseable timestamp.
    candidates.sort(
        key=lambda candidate: (
            candidate.timestamp or datetime.min,
            candidate.modified_at,
            candidate.path.name,
        ),
        reverse=True,
    )
    default_index = 0

    print(f'\nAvailable {label} artifacts:')
    for index, candidate in enumerate(candidates, start=1):
        default_marker = ' [default]' if index - 1 == default_index else ''
        modified = candidate.modified_at.strftime('%Y-%m-%d %H:%M:%S')
        print(
            f'  [{index}] {candidate.artifact_id}  '
            f'({candidate.path}, modified {modified}){default_marker}'
        )

    prompt = (
        f'Select {label} [1-{len(candidates)}] '
        f'or press Enter for default'
    )
    if timeout_seconds > 0:
        prompt += f' ({timeout_seconds}s timeout)'
    prompt += ': '

    response = _input_with_timeout(prompt, timeout_seconds).strip()
    if response:
        try:
            selected_index = int(response) - 1
        except ValueError:
            print(f'Invalid selection {response!r}; using default.')
            selected_index = default_index
        else:
            if selected_index not in range(len(candidates)):
                print(f'Invalid selection {response!r}; using default.')
                selected_index = default_index
    else:
        selected_index = default_index

    selected = candidates[selected_index]
    print(f'Selected {label}: {selected.artifact_id} ({selected.path})')
    return selected.artifact_id


def _find_candidates(
    directory: Path,
    pattern: str,
    id_prefix: str,
    id_suffix: str,
) -> list[ArtifactCandidate]:
    candidates: list[ArtifactCandidate] = []
    for path in directory.glob(pattern):
        if not path.is_file():
            continue
        artifact_id = _extract_id(path.name, id_prefix, id_suffix)
        if artifact_id is None:
            continue
        candidates.append(
            ArtifactCandidate(
                artifact_id=artifact_id,
                path=path,
                timestamp=_timestamp_from_id(artifact_id),
            )
        )
    return candidates


def _extract_id(name: str, prefix: str, suffix: str) -> str | None:
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    return name[len(prefix):-len(suffix)]


def _timestamp_from_id(artifact_id: str) -> datetime | None:
    match = _TIMESTAMP_RE.search(artifact_id)
    if not match:
        return None
    return datetime.strptime(match.group(), '%Y-%m-%d_%H-%M-%S')


def _input_with_timeout(prompt: str, timeout_seconds: int) -> str:
    if timeout_seconds <= 0:
        return input(prompt)

    if os.name == 'nt':
        if not sys.stdin.isatty():
            print(prompt)
            print('No interactive stdin available; using default.')
            return ''
        return _windows_input_with_timeout(prompt, timeout_seconds)

    print(prompt, end='', flush=True)
    try:
        readable, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    except (OSError, ValueError):
        print()
        print('Timed input is not available; using default.')
        return ''

    if readable:
        response = sys.stdin.readline()
        if response == '':
            print()
            print('No interactive stdin available; using default.')
        return response

    print()
    print('Selection timed out; using default.')
    return ''


def _windows_input_with_timeout(prompt: str, timeout_seconds: int) -> str:
    import msvcrt

    print(prompt, end='', flush=True)
    deadline = time.monotonic() + timeout_seconds
    chars: list[str] = []
    while time.monotonic() < deadline:
        if not msvcrt.kbhit():
            time.sleep(0.05)
            continue

        char = msvcrt.getwch()
        if char in ('\r', '\n'):
            print()
            return ''.join(chars)
        if char == '\b':
            if chars:
                chars.pop()
                print('\b \b', end='', flush=True)
            continue
        chars.append(char)
        print(char, end='', flush=True)

    print()
    print('Selection timed out; using default.')
    return ''
