"""Pricing engine using Monte Carlo simulation."""
import numpy as np
from .instruments import OptionType, AsianOption
from .numerical_greeks import NumericalGreeks

class MonteCarloEngine:
    """Pricing engine using Monte Carlo simulation."""
    
    def __init__(self, seed: int = None):
        """
        Initialize Monte Carlo engine.
        
        Args:
            seed (int): Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.numerical_greeks = NumericalGreeks(self)
    
    def calculate(self, option, market_data, 
                 num_simulations: int = 100000, time_steps: int = 100):
        """
        Calculate option price using Monte Carlo simulation.
        
        Args:
            option: Option object
            market_data (MarketData): Market data container
            num_simulations (int): Number of simulation paths
            time_steps (int): Number of time steps in each path
            
        Returns:
            dict: Dictionary containing price and standard error
        """
        S = option.underlying_price
        K = option.strike_price
        T = option.expiration_time
        r = market_data.risk_free_rate
        sigma = market_data.volatility
        q = market_data.dividend_yield
        dt = T / time_steps
        
        # For Asian options, we need to track the average price
        is_asian = isinstance(option, AsianOption)
        
        # Initialize asset paths
        asset_paths = np.zeros((num_simulations, time_steps + 1))
        asset_paths[:, 0] = S
        
        # For Asian options, we need to know which time steps correspond to averaging dates
        if is_asian:
            averaging_indices = [int(date / dt) for date in option.averaging_dates if date <= T]
            averaging_indices = list(set(averaging_indices))
            averaging_indices.sort()
        
        # Generate random numbers
        z = np.random.standard_normal((num_simulations, time_steps))
        
        # Simulate asset paths
        for t in range(1, time_steps + 1):
            asset_paths[:, t] = asset_paths[:, t - 1] * np.exp(
                (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z[:, t - 1]
            )
        
        # Calculate payoff
        if is_asian:
            # For Asian options, use average price at specified dates
            averaging_prices = asset_paths[:, averaging_indices]
            # Calculate average for each path
            avg_prices = np.mean(averaging_prices, axis=1)
            # Calculate payoff for each path
            payoffs = np.zeros(num_simulations)
            for i in range(num_simulations):
                payoffs[i] = option.payoff(avg_prices[i])
        else:
            # For other options, use final price
            final_prices = asset_paths[:, -1]
            payoffs = np.zeros(num_simulations)
            for i in range(num_simulations):
                payoffs[i] = option.payoff(final_prices[i])
        
        # Discount payoffs to present value
        price = np.exp(-r * T) * np.mean(payoffs)
        
        # Calculate standard error
        std_error = np.std(payoffs) / np.sqrt(num_simulations)
        
        return {
            'price': price,
            'standard_error': std_error
        }
    
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
