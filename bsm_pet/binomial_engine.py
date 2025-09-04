"""Pricing engine using binomial tree model."""
import numpy as np
from .instruments import OptionType, ExerciseType
from .numerical_greeks import NumericalGreeks

class BinomialEngine:
    """Pricing engine using binomial tree model."""
    
    def __init__(self):
        self.numerical_greeks = NumericalGreeks(self)
    
    def calculate(self, option, market_data, steps=1000):
        """
        Calculate option price using binomial tree.
        
        Args:
            option: Option object
            market_data (MarketData): Market data container
            steps (int): Number of steps in binomial tree
            
        Returns:
            dict: Dictionary containing price
        """
        S = option.underlying_price
        K = option.strike_price
        T = option.expiration_time
        r = market_data.risk_free_rate
        sigma = market_data.volatility
        q = market_data.dividend_yield
        dt = T / steps
        
        # Up and down factors
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        
        # Risk-neutral probability
        p = (np.exp((r - q) * dt) - d) / (u - d)
        
        # Initialize asset prices at maturity
        asset_prices = np.zeros(steps + 1)
        for i in range(steps + 1):
            asset_prices[i] = S * (u ** (steps - i)) * (d ** i)
        
        # Initialize option values at maturity
        option_values = np.zeros(steps + 1)
        for i in range(steps + 1):
            option_values[i] = option.payoff(asset_prices[i])
        
        # Step backwards through the tree
        for j in range(steps - 1, -1, -1):
            current_time = j * dt
            
            for i in range(j + 1):
                # Calculate continuation value
                continuation_value = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
                
                # Calculate exercise value
                current_asset_price = S * (u ** (j - i)) * (d ** i)
                exercise_value = option.payoff(current_asset_price)
                
                # For European options, always use continuation value
                if option.exercise_type == ExerciseType.EUROPEAN:
                    option_values[i] = continuation_value
                
                # For American options, choose maximum of exercise and continuation
                elif option.exercise_type == ExerciseType.AMERICAN:
                    option_values[i] = max(exercise_value, continuation_value)
                
                # For Bermudan options, check if current time is an exercise date
                elif option.exercise_type == ExerciseType.BERMUDAN:
                    is_exercise_date = any(abs(current_time - date) < dt/2 for date in option.exercise_dates)
                    
                    if is_exercise_date:
                        option_values[i] = max(exercise_value, continuation_value)
                    else:
                        option_values[i] = continuation_value
        
        return {'price': option_values[0]}
    
    def calculate_greeks(self, option, market_data):
        """
        Calculate Greeks using numerical methods.
        
        Args:
            option: Option object
            market_data (MarketData): Market data container
            
        Returns:
            dict: Dictionary of Greek values
        """
        return self.numerical_greeks.calculate_all(option, market_data)