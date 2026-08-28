"""GGUF header reading — re-exported from ggufram, the published package.

ggufram (https://github.com/JesusMonjeGonzalez/ggufram) carries the
pure-stdlib reader and the unified-memory arithmetic so any tool can reuse
them; Hearthia is one consumer. Kept as a shim so internal imports and the
RAM budget gate stay stable.
"""

from ggufram.gguf import RamProfile, model_ram_profile, read_metadata

__all__ = ["RamProfile", "model_ram_profile", "read_metadata"]
