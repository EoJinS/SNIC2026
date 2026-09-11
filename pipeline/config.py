"""
Shared constants used across multiple pipeline scripts. Kept dependency-free
(no sionna/mitsuba import) so scripts that don't need ray tracing -- e.g.
generate_dataset.py, EM fitting -- don't have to pull in the GPU stack.
"""

# Default TX antenna count (M), overridable per-run via generate_channels.py's
# --m flag. Output filenames always encode the actual M/F used (see each
# script's docstring), so these are just the defaults for a quick/small run.
DEFAULT_M_TX = 16
NUM_SUBCARRIERS = 64
SUBCARRIER_SPACING = 240e3   # Hz
