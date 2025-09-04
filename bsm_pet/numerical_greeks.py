"""Numerical calculation of option Greeks using finite difference methods."""
import numpy as np
from .market import MarketData
from .instruments import Option

class NumericalGreeks:
    """
    Class for numerical calculation of option Greeks.
    
    Supports calculation of all major Greeks for any pricing engine.
    """
    
    def __init__(self, engine, h=0.001):
        """
        Initialize numerical Greeks calculator.
        
        Args:
            engine: Pricing engine to use
            h (float): Step size for numerical differentiation
        """
        self.engine = engine
        self.h = h
    
    def calculate_all(self, option, market_data):
        """
        Calculate all Greeks numerically.
        
        Args:
            option (Option): Option object
            market_data (MarketData): Market data container
            
        Returns:
            dict: Dictionary of Greek values
        """
        return {
            'delta': self.calculate_delta(option, market_data),
            'gamma': self.calculate_gamma(option, market_data),
            'theta': self.calculate_theta(option, market_data),
            'vega': self.calculate_vega(option, market_data),
            'rho': self.calculate_rho(option, market_data)
        }
    
    def calculate_delta(self, option, market_data):
        """Calculate delta numerically."""
        original_price = option.underlying_price
        
        # Price with increased underlying
        option.underlying_price = original_price * (1 + self.h)
        price_up = self.engine.calculate(option, market_data)['price']
        
        # Price with decreased underlying
        option.underlying_price = original_price * (1 - self.h)
        price_down = self.engine.calculate(option, market_data)['price']
        
        # Reset underlying price
        option.underlying_price = original_price
        
        return (price_up - price_down) / (2 * original_price * self.h)
    
    def calculate_gamma(self, option, market_data):
        """Calculate gamma numerically."""
        original_price = option.underlying_price
        
        # Price at original underlying
        price_original = self.engine.calculate(option, market_data)['price']
        
        # Price with increased underlying
        option.underlying_price = original_price * (1 + self.h)
        price_up = self.engine.calculate(option, market_data)['price']
        
        # Price with decreased underlying
        option.underlying_price = original_price * (1 - self.h)
        price_down = self.engine.calculate(option, market_data)['price']
        
        # Reset underlying price
        option.underlying_price = original_price
        
        return (price_up - 2 * price_original + price_down) / (original_price * self.h) ** 2
    
    def calculate_theta(self, option, market_data):
        """Calculate theta numerically (per day)."""
        original_time = option.expiration_time
        one_day = 1/365
        
        # Price with decreased time (one day less)
        option.expiration_time = original_time - one_day
        price_time_dec = self.engine.calculate(option, market_data)['price']
        
        # Reset time
        option.expiration_time = original_time
        
        # Get original price
        price_original = self.engine.calculate(option, market_data)['price']
        
        return (price_time_dec - price_original) / one_day
    
    def calculate_vega(self, option, market_data):
        """Calculate vega numerically (per 1% change in volatility)."""
        original_vol = market_data.volatility
        
        # Price with increased volatility
        market_data.volatility = original_vol * (1 + self.h)
        price_vol_up = self.engine.calculate(option, market_data)['price']
        
        # Price with decreased volatility
        market_data.volatility = original_vol * (1 - self.h)
        price_vol_down = self.engine.calculate(option, market_data)['price']
        
        # Reset volatility
        market_data.volatility = original_vol
        
        # Get original price
        price_original = self.engine.calculate(option, market_data)['price']
        
        return (price_vol_up - price_vol_down) / (2 * original_vol * self.h * 100)
    
    def calculate_rho(self, option, market_data):
        """Calculate rho numerically (per 1% change in interest rate)."""
        original_rate = market_data.risk_free_rate
        
        # Price with increased rate
        market_data.risk_free_rate = original_rate * (1 + self.h)
        price_rate_up = self.engine.calculate(option, market_data)['price']
        
        # Price with decreased rate
        market_data.risk_free_rate = original_rate * (1 - self.h)
        price_rate_down = self.engine.calculate(option, market_data)['price']
        
        # Reset rate
        market_data.risk_free_rate = original_rate
        
        return (price_rate_up - price_rate_down) / (2 * original_rate * self.h * 100)