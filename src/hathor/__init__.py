"""hathor: Multi-agent system for generating galaxy formation hypotheses and creating relevant plots from RAMSES simulation data."""
__version__ = "0.0.0"
__author__ = "Autumn Pearce, Yevhen Kylivnyk"


from .hathor import Hathor
from .agents import BrainstormerAgent
__all__ = ["Hathor","BrainstormerAgent"]