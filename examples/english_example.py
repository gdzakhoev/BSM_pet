"""
BSM_pet Library

This example demonstrates all functions available in the BSM_pet library
including option pricing, Greeks calculation, implied volatility,
risk metrics, and various option types.
"""

import numpy as np
from bsm_pet import (
    MarketData, EuropeanOption, AmericanOption, BermudanOption, AsianOption,
    OptionType, ExerciseType, AnalyticalEngine, BinomialEngine,
    MonteCarloEngine, LongstaffSchwartzEngine,
    calculate_implied_volatility, RiskMetrics
)

def demonstrate_all_option_types():
    """Demonstrate pricing for all option types."""
    print("=== All Option Types Pricing ===")
    
    # Market data
    market_data = MarketData(
        risk_free_rate=0.05,    # 5% annual interest rate
        volatility=0.2,         # 20% annual volatility
        dividend_yield=0.01     # 1% annual dividend yield
    )
    
    # Create different option types with same parameters
    S = 100.0  # Underlying price
    K = 100.0  # Strike price
    T = 1.0    # Time to expiration (years)
    
    # European option
    european_call = EuropeanOption(K, T, OptionType.CALL)
    european_call.underlying_price = S
    
    # American option
    american_call = AmericanOption(K, T, OptionType.CALL)
    american_call.underlying_price = S
    
    # Bermudan option (exercise quarterly)
    bermudan_call = BermudanOption(K, T, OptionType.CALL, [0.25, 0.5, 0.75, 1.0])
    bermudan_call.underlying_price = S
    
    # Asian option (average at specific dates)
    asian_call = AsianOption(K, T, OptionType.CALL, [0.5, 0.75, 1.0])
    asian_call.underlying_price = S
    
    # Price using appropriate engines
    analytical_engine = AnalyticalEngine()
    binomial_engine = BinomialEngine()
    mc_engine = MonteCarloEngine(seed=42)
    ls_engine = LongstaffSchwartzEngine()
    
    # European option (analytical)
    european_result = analytical_engine.calculate(european_call, market_data)
    print(f"European Call: ${european_result['price']:.4f}")
    
    # American option (binomial)
    american_result = binomial_engine.calculate(american_call, market_data, steps=1000)
    print(f"American Call: ${american_result['price']:.4f}")
    
    # Bermudan option (binomial)
    bermudan_result = binomial_engine.calculate(bermudan_call, market_data, steps=1000)
    print(f"Bermudan Call: ${bermudan_result['price']:.4f}")
    
    # Asian option (Monte Carlo)
    asian_result = mc_engine.calculate(asian_call, market_data, num_simulations=50000)
    print(f"Asian Call: ${asian_result['price']:.4f} ± {asian_result['standard_error']:.4f}")
    
    # American option (Longstaff-Schwartz)
    american_ls_result = ls_engine.calculate(american_call, market_data, num_simulations=10000)
    print(f"American Call (LS): ${american_ls_result['price']:.4f}")
    
    print()

def demonstrate_all_greeks():
    """Demonstrate calculation of all Greeks."""
    print("=== All Greeks Calculation ===")
    
    # Market data
    market_data = MarketData(
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.01
    )
    
    # Create a European call option
    option = EuropeanOption(
        strike_price=100.0,
        expiration_time=1.0,
        option_type=OptionType.CALL
    )
    option.underlying_price = 100.0
    
    # Calculate all Greeks using analytical engine
    analytical_engine = AnalyticalEngine()
    result = analytical_engine.calculate(option, market_data)
    
    print("European Call Option Greeks:")
    print(f"Price: ${result['price']:.4f}")
    print(f"Delta: {result['delta']:.4f}")
    print(f"Gamma: {result['gamma']:.6f}")
    print(f"Theta: {result['theta']:.4f} per day")
    print(f"Vega: {result['vega']:.4f} per 1% vol change")
    print(f"Rho: {result['rho']:.4f} per 1% rate change")
    
    # Calculate numerical Greeks for comparison
    binomial_engine = BinomialEngine()
    numerical_greeks = binomial_engine.calculate_greeks(option, market_data)
    
    print("\nNumerical Greeks (Binomial):")
    print(f"Delta: {numerical_greeks['delta']:.4f}")
    print(f"Gamma: {numerical_greeks['gamma']:.6f}")
    print(f"Theta: {numerical_greeks['theta']:.4f} per day")
    print(f"Vega: {numerical_greeks['vega']:.4f} per 1% vol change")
    print(f"Rho: {numerical_greeks['rho']:.4f} per 1% rate change")
    
    print()

def demonstrate_implied_volatility():
    """Demonstrate implied volatility calculation."""
    print("=== Implied Volatility Calculation ===")
    
    # Market data
    market_data = MarketData(
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.01
    )
    
    # Create a European call option
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
    market_price = fair_price  # Assume market price equals fair price
    implied_vol = calculate_implied_volatility(option, market_price, market_data)
    
    print(f"Option with 20% volatility has fair price: ${fair_price:.4f}")
    print(f"If market price is ${market_price:.4f}, implied volatility is: {implied_vol:.4f}")
    
    # Test with different market prices
    for premium in [0.5, 1.0, 2.0]:
        test_price = fair_price + premium
        test_implied_vol = calculate_implied_volatility(option, test_price, market_data)
        print(f"Market price ${test_price:.4f} → Implied volatility: {test_implied_vol:.4f}")
    
    print()

def demonstrate_risk_metrics():
    """Demonstrate all risk metrics calculations."""
    print("=== All Risk Metrics Calculation ===")
    
    # Generate sample returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)  # Mean return 0.1%, std 2%
    
    # Calculate all risk metrics
    var_95 = RiskMetrics.calculate_value_at_risk(returns, 0.95)
    var_99 = RiskMetrics.calculate_value_at_risk(returns, 0.99)
    es_95 = RiskMetrics.calculate_expected_shortfall(returns, 0.95)
    es_99 = RiskMetrics.calculate_expected_shortfall(returns, 0.99)
    sharpe = RiskMetrics.calculate_sharpe_ratio(returns, 0.05/252)
    sortino = RiskMetrics.calculate_sortino_ratio(returns, 0.05/252, 0.0)
    max_dd = RiskMetrics.calculate_max_drawdown(returns)
    
    print(f"Value at Risk (95%): {var_95:.6f}")
    print(f"Value at Risk (99%): {var_99:.6f}")
    print(f"Expected Shortfall (95%): {es_95:.6f}")
    print(f"Expected Shortfall (99%): {es_99:.6f}")
    print(f"Sharpe Ratio: {sharpe:.6f}")
    print(f"Sortino Ratio: {sortino:.6f}")
    print(f"Maximum Drawdown: {max_dd:.6f}")
    
    # Interpretation
    print("\nInterpretation:")
    print(f"• Sortino ratio ({sortino:.3f}) is higher than Sharpe ratio ({sharpe:.3f})")
    print("  indicating more upside volatility than downside volatility")
    print(f"• 95% VaR of {var_95:.4f} means we expect to lose more than this")
    print("  amount only 5% of the time")
    print(f"• Maximum drawdown of {max_dd:.2%} shows the worst peak-to-trough decline")
    
    print()

def demonstrate_put_options():
    """Demonstrate put option pricing."""
    print("=== Put Option Pricing ===")
    
    # Market data
    market_data = MarketData(
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.01
    )
    
    # Create European put option
    put_option = EuropeanOption(
        strike_price=100.0,
        expiration_time=1.0,
        option_type=OptionType.PUT
    )
    put_option.underlying_price = 100.0
    
    # Calculate price and Greeks
    analytical_engine = AnalyticalEngine()
    result = analytical_engine.calculate(put_option, market_data)
    
    print("European Put Option:")
    print(f"Price: ${result['price']:.4f}")
    print(f"Delta: {result['delta']:.4f}")
    print(f"Gamma: {result['gamma']:.6f}")
    print(f"Theta: {result['theta']:.4f} per day")
    print(f"Vega: {result['vega']:.4f} per 1% vol change")
    print(f"Rho: {result['rho']:.4f} per 1% rate change")
    
    # Compare with call option
    call_option = EuropeanOption(
        strike_price=100.0,
        expiration_time=1.0,
        option_type=OptionType.CALL
    )
    call_option.underlying_price = 100.0
    
    call_result = analytical_engine.calculate(call_option, market_data)
    
    # Verify put-call parity
    put_call_parity = result['price'] - call_result['price'] + 100.0 - 100.0 * np.exp(-0.05 * 1.0)
    print(f"\nPut-Call Parity Check: {put_call_parity:.6f} (should be close to 0)")
    
    print()

def demonstrate_different_strikes():
    """Demonstrate option pricing for different strike prices."""
    print("=== Different Strike Prices ===")
    
    # Market data
    market_data = MarketData(
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.01
    )
    
    # Create options with different strikes
    strikes = [90.0, 100.0, 110.0]  # ITM, ATM, OTM
    expiration_time = 1.0
    underlying_price = 100.0
    
    analytical_engine = AnalyticalEngine()
    
    print("Call Option Prices for Different Strikes:")
    for strike in strikes:
        option = EuropeanOption(strike, expiration_time, OptionType.CALL)
        option.underlying_price = underlying_price
        result = analytical_engine.calculate(option, market_data)
        moneyness = "ITM" if strike < underlying_price else "ATM" if strike == underlying_price else "OTM"
        print(f"K={strike:.1f} ({moneyness}): ${result['price']:.4f}")
    
    print("\nPut Option Prices for Different Strikes:")
    for strike in strikes:
        option = EuropeanOption(strike, expiration_time, OptionType.PUT)
        option.underlying_price = underlying_price
        result = analytical_engine.calculate(option, market_data)
        moneyness = "ITM" if strike > underlying_price else "ATM" if strike == underlying_price else "OTM"
        print(f"K={strike:.1f} ({moneyness}): ${result['price']:.4f}")
    
    print()

def demonstrate_time_decay():
    """Demonstrate option time decay (theta)."""
    print("=== Time Decay (Theta) ===")
    
    # Market data
    market_data = MarketData(
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.01
    )
    
    # Create options with different expiration times
    times = [1.0, 0.5, 0.25, 0.1]  # Years to expiration
    strike_price = 100.0
    underlying_price = 100.0
    
    analytical_engine = AnalyticalEngine()
    
    print("Call Option Theta for Different Expiration Times:")
    for time in times:
        option = EuropeanOption(strike_price, time, OptionType.CALL)
        option.underlying_price = underlying_price
        result = analytical_engine.calculate(option, market_data)
        print(f"T={time:.2f} years: Theta = {result['theta']:.4f} per day")
    
    print("\nPut Option Theta for Different Expiration Times:")
    for time in times:
        option = EuropeanOption(strike_price, time, OptionType.PUT)
        option.underlying_price = underlying_price
        result = analytical_engine.calculate(option, market_data)
        print(f"T={time:.2f} years: Theta = {result['theta']:.4f} per day")
    
    print()

if __name__ == "__main__":
    print("BSM_pet Library - Complete Usage Example (English)")
    print("=" * 60)
    print()
    
    demonstrate_all_option_types()
    demonstrate_all_greeks()
    demonstrate_implied_volatility()
    demonstrate_risk_metrics()
    demonstrate_put_options()
    demonstrate_different_strikes()
    demonstrate_time_decay()
    
    print("All examples completed successfully!")
