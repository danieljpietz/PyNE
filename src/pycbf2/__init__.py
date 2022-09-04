from .type import NESystem, Link, LinkType
import numba
import warnings

warnings.simplefilter("ignore", numba.core.errors.NumbaPerformanceWarning)
warnings.simplefilter("ignore", numba.core.errors.NumbaExperimentalFeatureWarning)
