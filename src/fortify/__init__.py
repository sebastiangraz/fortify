"""fortify: adversarial anti-removal perturbations for visible watermarks."""

__version__ = "0.1.0"

# The δ wire format: uint8 = DELTA_OFFSET + round(δ · 255). Shared with the TS client.
DELTA_OFFSET = 128
