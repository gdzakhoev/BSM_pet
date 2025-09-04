"""Pricing engine for American options using Longstaff-Schwartz algorithm."""
import numpy as np
from sklearn.linear_model import LinearRegression
from .instruments import OptionType

class LongstaffSchwartzEngine:
    """
    Pricing engine for American options using Longstaff-Schwartz algorithm.
    """
    
    def calculate(self, option, market_data, 
                 num_simulations=10000, time_steps=50):
        """
        Calculate American option price using Longstaff-Schwartz algorithm.
        """
        S = option.underlying_price
        K = option.strike_price
        T = option.expiration_time
        r = market_data.risk_free_rate
        sigma = market_data.volatility
        q = market_data.dividend_yield
        dt = T / time_steps
        
        # Generate asset paths
        asset_paths = np.zeros((num_simulations, time_steps + 1))
        asset_paths[:, 0] = S
        
        # Generate random numbers
        z = np.random.standard_normal((num_simulations, time_steps))
        
        # Simulate asset paths
        for t in range(1, time_steps + 1):
            asset_paths[:, t] = asset_paths[:, t - 1] * np.exp(
                (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z[:, t - 1]
            )
        
        # Initialize cash flows matrix
        cash_flows = np.zeros((num_simulations, time_steps + 1))
        
        # At expiration, cash flow is the payoff
        if option.option_type == OptionType.CALL:
            cash_flows[:, -1] = np.maximum(asset_paths[:, -1] - K, 0.0)
        else:
            cash_flows[:, -1] = np.maximum(K - asset_paths[:, -1], 0.0)
        
        # Work backwards through time
        for t in range(time_steps - 1, 0, -1):
            # Find paths that are in-the-money at time t
            if option.option_type == OptionType.CALL:
                in_the_money = asset_paths[:, t] > K
            else:
                in_the_money = asset_paths[:, t] < K
            
            if np.any(in_the_money):
                # Get discounted future cash flows for in-the-money paths
                discounted_cash_flows = np.exp(-r * dt) * cash_flows[:, t + 1]
                
                # Get current asset prices for in-the-money paths
                current_prices = asset_paths[in_the_money, t]
                
                # Regression of discounted cash flows on current prices
                X = current_prices.reshape(-1, 1)
                X2 = X * X
                Xs = np.column_stack([X, X2])
                Y = discounted_cash_flows[in_the_money]
                
                # Skip if not enough data points
                if len(Y) < 3:
                    continue
                
                # Fit quadratic polynomial
                try:
                    model = LinearRegression()
                    model.fit(Xs, Y)
                    continuation_values = model.predict(Xs)
                    
                    # Calculate exercise values
                    exercise_values = option.payoff(current_prices)
                    
                    # Decide whether to exercise
                    exercise = exercise_values > continuation_values
                    
                    # Update cash flows
                    cash_flows[in_the_money, t] = np.where(exercise, exercise_values, 0.0)
                    
                    # Set future cash flows to zero if we exercise now
                    cash_flows[in_the_money, t + 1:] = 0.0
                except:
                    # If regression fails, continue without exercise
                    continue
        
        # Discount all cash flows back to present value
        discount_factors = np.exp(-r * dt * np.arange(time_steps + 1))
        discounted_cash_flows = cash_flows * discount_factors
        price = np.mean(np.sum(discounted_cash_flows, axis=1))
        
        return {'price': price}
