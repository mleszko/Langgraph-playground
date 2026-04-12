"""Composition root for application wiring."""

from .container import AppContainer, build_default_container

__all__ = ["AppContainer", "build_default_container"]

