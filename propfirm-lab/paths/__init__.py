"""Equity-path generators that feed the account simulator."""

from paths.parametric import (
    TradeModel,
    generate_path,
    generate_paths,
)

__all__ = ["TradeModel", "generate_path", "generate_paths"]
