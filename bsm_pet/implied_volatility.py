"""Calculation of implied volatility for options."""
from scipy.optimize import brentq
from .market import MarketData

def calculate_implied_volatility(option, market_price, market_data, engine=None,
                                initial_guess=0.2, max_iter=100, tol=1e-6):
    """
    Calculate implied volatility for any option type.
    
    Args:
        option: Option object
        market_price (float): Observed market price of the option
        market_data (MarketData): Market data container
        engine: Pricing engine to use
        initial_guess (float): Initial guess for volatility
        max_iter (int): Maximum number of iterations
        tol (float): Tolerance for convergence
        
    Returns:
        float: Implied volatility
        
    Raises:
        ValueError: If implied volatility calculation fails
    """
    if engine is None:
        # Use analytical engine for European options, binomial for others
        from .instruments import ExerciseType
        if option.exercise_type == ExerciseType.EUROPEAN:
            from .analytical_engine import AnalyticalEngine
            engine = AnalyticalEngine()
        else:
            from .binomial_engine import BinomialEngine
            engine = BinomialEngine()
    
    # Define the function to solve (price difference)
    def price_difference(sigma):
        new_market_data = MarketData(
            risk_free_rate=market_data.risk_free_rate,
            volatility=sigma,
            dividend_yield=market_data.dividend_yield
        )
        result = engine.calculate(option, new_market_data)
        calculated_price = result['price'] if isinstance(result, dict) else result
        return calculated_price - market_price
    
    # Find the root of the function
    try:
        implied_vol = brentq(price_difference, 0.001, 5.0,
                            maxiter=max_iter, xtol=tol)
        return implied_vol
    except ValueError:
        try:
            # Try with wider bounds
            implied_vol = brentq(price_difference, 0.0001, 10.0,
                                maxiter=max_iter, xtol=tol)
            return implied_vol
        except ValueError:
            raise ValueError("Implied volatility calculation failed. Check if market price is within possible range.")