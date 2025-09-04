"""Option contract definitions."""
from enum import Enum
from typing import List, Optional
import numpy as np

class OptionType(Enum):
    """Type of option contract."""
    CALL = 1
    PUT = 2

class ExerciseType(Enum):
    """Exercise style of option contract."""
    EUROPEAN = 1
    AMERICAN = 2
    BERMUDAN = 3
    ASIAN = 4

class Option:
    """Base class for all option contracts."""
    
    def __init__(self, strike_price, expiration_time, option_type,
                 exercise_type, underlying_price=None):
        """
        Initialize option contract.
        
        Args:
            strike_price (float): Option strike price
            expiration_time (float): Time to expiration in years
            option_type (OptionType): CALL or PUT
            exercise_type (ExerciseType): Exercise style
            underlying_price (float): Current underlying price
        """
        self.strike_price = strike_price
        self.expiration_time = expiration_time
        self.option_type = option_type
        self.exercise_type = exercise_type
        self.underlying_price = underlying_price
        
    def payoff(self, spot_price):
        """
        Calculate option payoff at expiration.
        
        Args:
            spot_price (float): Price of underlying asset
            
        Returns:
            float: Payoff value
        """
        if self.option_type == OptionType.CALL:
            return max(spot_price - self.strike_price, 0)
        return max(self.strike_price - spot_price, 0)
            
    def __repr__(self):
        return f"Option(K={self.strike_price}, T={self.expiration_time}, type={self.option_type}, exercise={self.exercise_type})"

class EuropeanOption(Option):
    """European option (exercise only at expiration)."""
    
    def __init__(self, strike_price, expiration_time, option_type,
                 underlying_price=None):
        super().__init__(strike_price, expiration_time, option_type,
                         ExerciseType.EUROPEAN, underlying_price)

class AmericanOption(Option):
    """American option (exercise at any time before expiration)."""
    
    def __init__(self, strike_price, expiration_time, option_type,
                 underlying_price=None):
        super().__init__(strike_price, expiration_time, option_type,
                         ExerciseType.AMERICAN, underlying_price)

class BermudanOption(Option):
    """Bermudan option (exercise at specific dates)."""
    
    def __init__(self, strike_price, expiration_time, option_type,
                 exercise_dates, underlying_price=None):
        super().__init__(strike_price, expiration_time, option_type,
                         ExerciseType.BERMUDAN, underlying_price)
        self.exercise_dates = exercise_dates

class AsianOption(Option):
    """Asian option (payoff depends on average price)."""
    
    def __init__(self, strike_price, expiration_time, option_type,
                 averaging_dates, underlying_price=None):
        super().__init__(strike_price, expiration_time, option_type,
                         ExerciseType.ASIAN, underlying_price)
        self.averaging_dates = averaging_dates
        
    def payoff(self, spot_prices):
        """
        Calculate payoff for Asian option.
        
        Args:
            spot_prices (List[float]): List of spot prices
            
        Returns:
            float: Payoff value
        """
        avg_price = np.mean(spot_prices)
        if self.option_type == OptionType.CALL:
            return max(avg_price - self.strike_price, 0)
        return max(self.strike_price - avg_price, 0)