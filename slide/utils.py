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
from collections.abc import Callable, Iterable
import numpy as np

def get_repo_root() -> Path:
    """Return the absolute path to the repository root.

    Returns:
    - Path
        The parent directory of the ``slide`` package. All default data directories are resolved relative to this path.
    """
    return Path(__file__).resolve().parents[1]

def resolve_data_dir(env_var: str, default_name: str, *, create: bool=True) -> Path:
    """Resolve a data directory from an environment override or repo default.

    Parameters:
    - env_var: str
        Name of an environment variable that may contain an absolute or user-relative path.
    - default_name: str
        Repository-relative directory name to use when ``env_var`` is not set.
    - create: bool
        If true, create the directory and parents before returning.

    Returns:
    - Path
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

    Parameters:
    - value: float | int
        Integer-like or floating-point value to encode.

    Returns:
    - str
        A short decimal string without unnecessary trailing zeroes.
    """
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value = float(value)
    text = f'{value:.6g}'
    return text.rstrip('0').rstrip('.') if '.' in text else text

def parameterized_filename(kind: str, **params: object) -> str:
    """Build a descriptive ``.pkl`` filename from run-defining parameters.

    Parameters:
    - kind: str
        Product family, such as ``nk_decay`` or ``empirical_strategy_GB1``.
    - **params: object
        Ordered parameter names and values to append to the filename. ``None``
        values are skipped; lists and ranges are compressed as ``first-last``
        when they contain more than two entries.

    Returns:
    - str
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

def load_pickle(path: str | Path) -> object:
    """Load a pickle object from an explicit path.

    Parameters:
    - path: str | Path
        File path to read.

    Returns:
    - object
        The deserialized Python object.
    """
    with Path(path).open('rb') as handle:
        return pickle.load(handle)

def save_pickle(obj: object, path: str | Path) -> Path:
    """Serialize an object to pickle, creating parent directories if needed.

    Parameters:
    - obj: object
        Python object to serialize.
    - path: str | Path
        Destination file path.

    Returns:
    - Path
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

    Parameters:
    - filename: str
        Figure stem or filename.
    - fmt: str
        Output format such as ``pdf``, ``eps``, or ``png``.

    Returns:
    - Path
        A path under ``figures/<fmt>/`` with the requested extension.
    """
    fmt = fmt.lower().lstrip('.')
    stem = Path(filename).stem
    return get_figures_dir() / fmt / f'{stem}.{fmt}'

def load_raw(filename: str) -> object:
    """Load a raw simulation product by filename from ``raw_data``."""
    return load_pickle(raw_path(filename))

def save_raw(obj: object, filename: str) -> Path:
    """Save a raw simulation product by filename into ``raw_data``."""
    return save_pickle(obj, raw_path(filename))

def load_processed(filename: str) -> object:
    """Load a processed plotting input by filename from ``processed_data``."""
    return load_pickle(processed_path(filename))

def save_processed(obj: object, filename: str) -> Path:
    """Save a processed plotting input by filename into ``processed_data``."""
    return save_pickle(obj, processed_path(filename))

def install_multi_format_savefig(plt: object, save_type_list: Iterable[str]) -> Callable[..., object]:
    """Patch ``matplotlib.pyplot.savefig`` to write each figure in all formats.

    Parameters:
    - plt: object
        Imported ``matplotlib.pyplot`` module.
    - save_type_list: Iterable[str]
        Iterable of formats to write, for example ``["pdf", "eps", "png"]``.

    Returns:
    - Callable[..., object]
        The original ``plt.savefig`` callable so callers can restore it if needed. Calls whose path begins with ``figures`` are redirected to ``figures/<format>/<stem>.<format>`` for every requested format.
    """
    original_savefig = plt.savefig
    formats = [fmt.lower().lstrip('.') for fmt in save_type_list]

    def savefig(path: str | Path, *args: object, **kwargs: object) -> None:
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
