"""Risk metrics calculation for financial instruments."""
import numpy as np

class RiskMetrics:
    """Class for calculating various risk metrics."""
    
    @staticmethod
    def calculate_value_at_risk(returns, confidence_level=0.95):
        """
        Calculate Value at Risk (VaR) for given returns.
        
        Args:
            returns (np.ndarray): Array of returns
            confidence_level (float): Confidence level
            
        Returns:
            float: Value at Risk
        """
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def calculate_expected_shortfall(returns, confidence_level=0.95):
        """
        Calculate Expected Shortfall (ES) for given returns.
        
        Args:
            returns (np.ndarray): Array of returns
            confidence_level (float): Confidence level
            
        Returns:
            float: Expected Shortfall
        """
        var = RiskMetrics.calculate_value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_beta(option_returns, market_returns):
        """
        Calculate beta of option returns against market returns.
        
        Args:
            option_returns (np.ndarray): Option returns
            market_returns (np.ndarray): Market returns
            
        Returns:
            float: Beta coefficient
        """
        covariance = np.cov(option_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        return covariance / market_variance
    
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
        """
        Calculate Sharpe ratio for given returns.
        
        Args:
            returns (np.ndarray): Array of returns
            risk_free_rate (float): Risk-free rate
            
        Returns:
            float: Sharpe ratio
        """
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns)
    
    @staticmethod
    def calculate_sortino_ratio(returns, risk_free_rate=0.0, target_return=0.0):
        """
        Calculate Sortino ratio for given returns.
        
        Args:
            returns (np.ndarray): Array of returns
            risk_free_rate (float): Risk-free rate
            target_return (float): Target return
            
        Returns:
            float: Sortino ratio
        """
        excess_returns = returns - risk_free_rate
        downside_returns = returns[returns < target_return] - target_return
        downside_risk = np.sqrt(np.mean(downside_returns ** 2))
        return np.mean(excess_returns) / downside_risk
    
    @staticmethod
    def calculate_max_drawdown(returns):
        """
        Calculate maximum drawdown for given returns.
        
        Args:
            returns (np.ndarray): Array of returns
            
        Returns:
            float: Maximum drawdown
        """
        cumulative_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        return np.max(drawdown)