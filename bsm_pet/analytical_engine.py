"""Analytical pricing engine using Black-Scholes-Merton formula."""
from .utils import norm_cdf, norm_pdf
from .market import MarketData
from .instruments import OptionType
import numpy as np

class AnalyticalEngine:
    """Pricing engine using analytical Black-Scholes-Merton formula."""
    
    def calculate(self, option, market_data):
        """
        Calculate option price and Greeks using BSM formula.
        
        Args:
            option: Option object
            market_data (MarketData): Market data
            
        Returns:
            dict: Price and Greeks
        """
        S = option.underlying_price
        K = option.strike_price
        T = option.expiration_time
        r = market_data.risk_free_rate
        sigma = market_data.volatility
        q = market_data.dividend_yield
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option.option_type == OptionType.CALL:
            price = S * np.exp(-q * T) * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm_cdf(-d2) - S * np.exp(-q * T) * norm_cdf(-d1)
        
        delta = self._calculate_delta(option, market_data, d1)
        gamma = self._calculate_gamma(option, market_data, d1)
        theta = self._calculate_theta(option, market_data, d1, d2)
        vega = self._calculate_vega(option, market_data, d1)
        rho = self._calculate_rho(option, market_data, d1, d2)
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def _calculate_delta(self, option, market_data, d1=None):
        """Calculate option delta."""
        S = option.underlying_price
        K = option.strike_price
        T = option.expiration_time
        r = market_data.risk_free_rate
        sigma = market_data.volatility
        q = market_data.dividend_yield
        
        if d1 is None:
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        if option.option_type == OptionType.CALL:
            return np.exp(-q * T) * norm_cdf(d1)
        return np.exp(-q * T) * (norm_cdf(d1) - 1)
    
    def _calculate_gamma(self, option, market_data, d1=None):
        """Calculate option gamma."""
        S = option.underlying_price
        K = option.strike_price
        T = option.expiration_time
        r = market_data.risk_free_rate
        sigma = market_data.volatility
        q = market_data.dividend_yield
        
        if d1 is None:
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        return np.exp(-q * T) * norm_pdf(d1) / (S * sigma * np.sqrt(T))
    
    def _calculate_vega(self, option, market_data, d1=None):
        """Calculate option vega."""
        S = option.underlying_price
        K = option.strike_price
        T = option.expiration_time
        r = market_data.risk_free_rate
        sigma = market_data.volatility
        q = market_data.dividend_yield
        
        if d1 is None:
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        return S * np.exp(-q * T) * norm_pdf(d1) * np.sqrt(T) / 100
    
    def _calculate_theta(self, option, market_data, d1=None, d2=None):
        """Calculate option theta."""
        S = option.underlying_price
        K = option.strike_price
        T = option.expiration_time
        r = market_data.risk_free_rate
        sigma = market_data.volatility
        q = market_data.dividend_yield
        
        if d1 is None:
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if d2 is None:
            d2 = d1 - sigma * np.sqrt(T)
        
        if option.option_type == OptionType.CALL:
            theta = (-S * norm_pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T)) -
                     r * K * np.exp(-r * T) * norm_cdf(d2) +
                     q * S * np.exp(-q * T) * norm_cdf(d1))
        else:
            theta = (-S * norm_pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T)) +
                     r * K * np.exp(-r * T) * norm_cdf(-d2) -
                     q * S * np.exp(-q * T) * norm_cdf(-d1))
        
        return theta / 365
    
    def _calculate_rho(self, option, market_data, d1=None, d2=None):
        """Calculate option rho."""
        S = option.underlying_price
        K = option.strike_price
        T = option.expiration_time
        r = market_data.risk_free_rate
        sigma = market_data.volatility
        q = market_data.dividend_yield
        
        if d1 is None:
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if d2 is None:
            d2 = d1 - sigma * np.sqrt(T)
        
        if option.option_type == OptionType.CALL:
            rho = K * T * np.exp(-r * T) * norm_cdf(d2)
        else:
            rho = -K * T * np.exp(-r * T) * norm_cdf(-d2)
        
        return rho / 100