"""Shared utility helpers for the refactored SLIDE notebooks.

This module contains small, non-scientific helpers used by the notebook-driven
pipeline: repository-relative paths, pickle I/O, descriptive filename creation,
and the multi-format figure saving hook. Keeping these here avoids spreading
filesystem and filename conventions across generation, processing, and plotting
code.
"""
from __future__ import annotations
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Iterable
import numpy as np

def get_repo_root() -> Path:
    """Return the absolute path to the repository root.

    Returns:
        The parent directory of the ``slide`` package. All default data
        directories are resolved relative to this path.
    """
    return Path(__file__).resolve().parents[1]

def resolve_data_dir(env_var: str, default_name: str, *, create: bool=True) -> Path:
    """Resolve a data directory from an environment override or repo default.

    Args:
        env_var: Name of an environment variable that may contain an absolute or
            user-relative path.
        default_name: Repository-relative directory name to use when
            ``env_var`` is not set.
        create: If true, create the directory and parents before returning.

    Returns:
        The resolved absolute directory path.
    """
    raw_value = os.getenv(env_var)
    path = Path(raw_value).expanduser() if raw_value else get_repo_root() / default_name
    path = path.resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path

def get_raw_data_dir(*, create: bool=True) -> Path:
    """Return the directory where raw simulation products are stored."""
    return resolve_data_dir('SLIDE_RAW_DATA_DIR', 'raw_data', create=create)

def get_processed_data_dir(*, create: bool=True) -> Path:
    """Return the directory where processed plotting inputs are stored."""
    return resolve_data_dir('SLIDE_PROCESSED_DATA_DIR', 'processed_data', create=create)

def get_figures_dir(*, create: bool=True) -> Path:
    """Return the directory where notebook figures are written."""
    return resolve_data_dir('SLIDE_FIGURES_DIR', 'figures', create=create)

def get_landscape_arrays_dir() -> Path:
    """Return the repository directory containing empirical landscape arrays."""
    return get_repo_root() / 'landscape_arrays'

def get_other_data_dir() -> Path:
    """Return the repository directory containing small auxiliary data files."""
    return get_repo_root() / 'other_data'

def format_number(value: float | int) -> str:
    """Format numeric values compactly for stable descriptive filenames.

    Args:
        value: Integer-like or floating-point value to encode.

    Returns:
        A short decimal string without unnecessary trailing zeroes.
    """
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value = float(value)
    text = f'{value:.6g}'
    return text.rstrip('0').rstrip('.') if '.' in text else text

def parameterized_filename(kind: str, **params: Any) -> str:
    """Build a descriptive ``.pkl`` filename from run-defining parameters.

    Args:
        kind: Product family, such as ``nk_decay`` or ``empirical_strategy_GB1``.
        **params: Ordered parameter names and values to append to the filename.
            ``None`` values are skipped; lists/ranges are compressed as
            ``first-last`` when they contain more than two entries.

    Returns:
        A filename ending in ``.pkl``.
    """
    parts = [kind]
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, range)):
            if len(value) > 2:
                text = f'{value[0]}-{value[-1]}'
            else:
                text = '-'.join((format_number(v) for v in value))
        else:
            text = format_number(value) if isinstance(value, (float, int, np.integer, np.floating)) else str(value)
        parts.append(f'{key}{text}')
    return '_'.join(parts) + '.pkl'

def load_pickle(path: str | Path) -> Any:
    """Load a pickle object from an explicit path.

    Args:
        path: File path to read.

    Returns:
        The deserialized Python object.
    """
    with Path(path).open('rb') as handle:
        return pickle.load(handle)

def save_pickle(obj: Any, path: str | Path) -> Path:
    """Serialize an object to pickle, creating parent directories if needed.

    Args:
        obj: Python object to serialize.
        path: Destination file path.

    Returns:
        The resolved destination path as a ``Path`` object.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        pickle.dump(obj, handle)
    return path

def raw_path(filename: str) -> Path:
    """Return a path inside ``raw_data`` for a raw product filename."""
    return get_raw_data_dir() / filename

def processed_path(filename: str) -> Path:
    """Return a path inside ``processed_data`` for a processed product filename."""
    return get_processed_data_dir() / filename

def figure_path(filename: str, fmt: str) -> Path:
    """Return a figure path inside the format-specific figure subdirectory.

    Args:
        filename: Figure stem or filename.
        fmt: Output format such as ``pdf``, ``eps``, or ``png``.

    Returns:
        A path under ``figures/<fmt>/`` with the requested extension.
    """
    fmt = fmt.lower().lstrip('.')
    stem = Path(filename).stem
    return get_figures_dir() / fmt / f'{stem}.{fmt}'

def load_raw(filename: str) -> Any:
    """Load a raw simulation product by filename from ``raw_data``."""
    return load_pickle(raw_path(filename))

def save_raw(obj: Any, filename: str) -> Path:
    """Save a raw simulation product by filename into ``raw_data``."""
    return save_pickle(obj, raw_path(filename))

def load_processed(filename: str) -> Any:
    """Load a processed plotting input by filename from ``processed_data``."""
    return load_pickle(processed_path(filename))

def save_processed(obj: Any, filename: str) -> Path:
    """Save a processed plotting input by filename into ``processed_data``."""
    return save_pickle(obj, processed_path(filename))

def install_multi_format_savefig(plt: Any, save_type_list: Iterable[str]) -> Callable[..., Any]:
    """Patch ``matplotlib.pyplot.savefig`` to write each figure in all formats.

    Args:
        plt: Imported ``matplotlib.pyplot`` module.
        save_type_list: Iterable of formats to write, for example
            ``["pdf", "eps", "png"]``.

    Returns:
        The original ``plt.savefig`` callable so callers can restore it if
        needed. Calls whose path begins with ``figures`` are redirected to
        ``figures/<format>/<stem>.<format>`` for every requested format.
    """
    original_savefig = plt.savefig
    formats = [fmt.lower().lstrip('.') for fmt in save_type_list]

    def savefig(path: str | Path, *args: Any, **kwargs: Any) -> None:
        """Support the notebook-driven SLIDE analysis pipeline.

This helper is part of the refactored paper code. Its typed signature defines
which arrays, parameters, or file labels it accepts; callers use it from the
three notebooks rather than executing standalone scripts. The function has no
hidden notebook state and returns the explicit array, scalar, path, or summary
object consumed by downstream generation, processing, or visualisation cells."""
        path_obj = Path(path)
        if path_obj.parts and path_obj.parts[0] == 'figures' and path_obj.suffix:
            for fmt in formats:
                out_path = figure_path(path_obj.stem, fmt)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                original_savefig(out_path, *args, **kwargs)
            return None
        original_savefig(path, *args, **kwargs)
        return None
    plt.savefig = savefig
    return original_savefig
