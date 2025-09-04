"""BSM_pet - Comprehensive option pricing library."""

from .market import MarketData
from .instruments import (
    EuropeanOption, AmericanOption, BermudanOption, AsianOption,
    OptionType, ExerciseType
)
from .analytical_engine import AnalyticalEngine
from .binomial_engine import BinomialEngine
from .monte_carlo_engine import MonteCarloEngine
from .longstaff_schwartz_engine import LongstaffSchwartzEngine
from .implied_volatility import calculate_implied_volatility
from .risk_metrics import RiskMetrics

__version__ = "0.1.0"
__all__ = [
    'MarketData',
    'EuropeanOption', 'AmericanOption', 'BermudanOption', 'AsianOption',
    'OptionType', 'ExerciseType',
    'AnalyticalEngine', 'BinomialEngine', 'MonteCarloEngine',
    'LongstaffSchwartzEngine',
    'calculate_implied_volatility',
    'RiskMetrics'
]
