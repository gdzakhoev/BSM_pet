"""
BSM_pet Library Usage Example - English Version

This example demonstrates the basic usage of the BSM_pet library
for option pricing and risk analysis.
"""

import numpy as np
from bsm_pet import (
    MarketData, EuropeanOption, AmericanOption, AsianOption,
    OptionType, AnalyticalEngine, BinomialEngine, MonteCarloEngine,
    calculate_implied_volatility, RiskMetrics
)

def basic_option_pricing():
    """Basic option pricing example."""
    print("=== Basic Option Pricing ===")
    
    # Market data
    market_data = MarketData(
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.01
    )
    
    # Create a European call option
    call_option = EuropeanOption(
        strike_price=100.0,
        expiration_time=1.0,
        option_type=OptionType.CALL
    )
    call_option.underlying_price = 100.0
    
    # Price using analytical engine
    analytical_engine = AnalyticalEngine()
    result = analytical_engine.calculate(call_option, market_data)
    
    print(f"European Call Option:")
    print(f"  Price: ${result['price']:.4f}")
    print(f"  Delta: {result['delta']:.4f}")
    print(f"  Gamma: {result['gamma']:.6f}")
    print(f"  Theta: {result['theta']:.4f} per day")
    print(f"  Vega: {result['vega']:.4f} per 1% vol change")
    print(f"  Rho: {result['rho']:.4f} per 1% rate change")
    print()

def implied_volatility_example():
    """Implied volatility calculation example."""
    print("=== Implied Volatility Calculation ===")
    
    # Market data
    market_data = MarketData(
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.01
    )
    
    # Create option
    option = EuropeanOption(
        strike_price=100.0,
        expiration_time=1.0,
        option_type=OptionType.CALL
    )
    option.underlying_price = 100.0
    
    # Calculate fair price first
    analytical_engine = AnalyticalEngine()
    fair_price = analytical_engine.calculate(option, market_data)['price']
    
    # Calculate implied volatility from market price
    market_price = fair_price
    implied_vol = calculate_implied_volatility(option, market_price, market_data)
    
    print(f"Option with 20% volatility has fair price: ${fair_price:.4f}")
    print(f"If market price is ${market_price:.4f}, implied volatility is: {implied_vol:.4f}")
    print()

def american_option_example():
    """American option pricing example."""
    print("=== American Option Pricing ===")
    
    # Market data
    market_data = MarketData(
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.01
    )
    
    # Create American call option
    american_option = AmericanOption(
        strike_price=100.0,
        expiration_time=1.0,
        option_type=OptionType.CALL
    )
    american_option.underlying_price = 100.0
    
    # Price using binomial tree
    binomial_engine = BinomialEngine()
    result = binomial_engine.calculate(american_option, market_data, steps=1000)
    
    print(f"American Call Option Price: ${result['price']:.4f}")
    
    # Compare with European option
    european_option = EuropeanOption(
        strike_price=100.0,
        expiration_time=1.0,
        option_type=OptionType.CALL
    )
    european_option.underlying_price = 100.0
    
    analytical_engine = AnalyticalEngine()
    european_price = analytical_engine.calculate(european_option, market_data)['price']
    
    print(f"European Call Option Price: ${european_price:.4f}")
    print(f"Early Exercise Premium: ${result['price'] - european_price:.4f}")
    print()

def risk_metrics_example():
    """Risk metrics calculation example."""
    print("=== Risk Metrics Calculation ===")
    
    # Generate sample returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)
    
    # Calculate risk metrics
    var_95 = RiskMetrics.calculate_value_at_risk(returns, 0.95)
    es_95 = RiskMetrics.calculate_expected_shortfall(returns, 0.95)
    sharpe = RiskMetrics.calculate_sharpe_ratio(returns, 0.05/252)
    max_drawdown = RiskMetrics.calculate_max_drawdown(returns)
    
    print(f"Value at Risk (95%): {var_95:.6f}")
    print(f"Expected Shortfall (95%): {es_95:.6f}")
    print(f"Sharpe Ratio: {sharpe:.6f}")
    print(f"Maximum Drawdown: {max_drawdown:.6f}")
    print()

if __name__ == "__main__":
    print("BSM_pet Library Examples - English Version")
    print("=" * 50)
    print()
    
    basic_option_pricing()
    implied_volatility_example()
    american_option_example()
    risk_metrics_example()
    
    print("All examples completed successfully!")
