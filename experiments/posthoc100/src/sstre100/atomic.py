"""Minimal Volume-compatible checkpoint bytes; publication is verified separately.

The historical function name is retained to avoid changing callers. This writer
uses no filesystem rename or replacement. A final-path file is INVALID until
its existing publisher commit/readback/verified-marker sequence succeeds.
"""
import os
from pathlib import Path
import torch

def atomic_torch_save(value,destination):
    path=Path(destination)
    # O_EXCL via xb refuses existing files, including leftover failed writes.
    with path.open('xb') as stream:
        try:
            torch.save(value,stream)
            stream.flush();os.fsync(stream.fileno())
        except Exception as exc:
            exc.add_note('Unverified incomplete checkpoint, never use or retry: '+str(path))
            raise
    return path
