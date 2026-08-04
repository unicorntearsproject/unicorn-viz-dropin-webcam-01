"""Webcam drop-in package for unicorn-viz.

Core loads symbols by file path (``load_dropin_symbol('webcam-01/
webcam_overlay.py', 'WebcamSystem')``), never through this package — and
the previous eager relative re-export broke pytest's module import of the
drop-in tree. Kept import-free like every other drop-in ``__init__``.
"""
from __future__ import annotations
