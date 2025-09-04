"""Market data container for option pricing."""

class MarketData:
    """Container for market data with constant rates."""
    
    def __init__(self, risk_free_rate, volatility, dividend_yield=0.0):
        """
        Initialize market data.
        
        Args:
            risk_free_rate (float): Annual risk-free interest rate
            volatility (float): Annual volatility
            dividend_yield (float): Annual dividend yield
        """
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.dividend_yield = dividend_yield
        
    def __repr__(self):
        return f"MarketData(r={self.risk_free_rate}, sigma={self.volatility}, q={self.dividend_yield})"