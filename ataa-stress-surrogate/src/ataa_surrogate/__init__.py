"""Shared scientific implementation for the ATAA stress surrogate.

Submodules are intentionally not imported eagerly: tensor and metric consumers remain
NumPy-only, while model, inference and loss consumers explicitly opt into PyTorch.
"""
