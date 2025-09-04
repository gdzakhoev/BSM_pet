"""
Пример использования библиотеки BSM_pet

Этот пример демонстрирует основные возможности библиотеки BSM_pet
для оценки опционов и анализа рисков.
"""

import numpy as np
from bsm_pet import (
    MarketData, EuropeanOption, AmericanOption, AsianOption,
    OptionType, AnalyticalEngine, BinomialEngine, MonteCarloEngine,
    calculate_implied_volatility, RiskMetrics
)

def basic_option_pricing():
    """Базовый пример оценки опционов."""
    print("=== Базовая оценка опционов ===")
    
    # Рыночные данные
    market_data = MarketData(
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.01
    )
    
    # Создаем европейский опцион колл
    call_option = EuropeanOption(
        strike_price=100.0,
        expiration_time=1.0,
        option_type=OptionType.CALL
    )
    call_option.underlying_price = 100.0
    
    # Оценка с использованием аналитического движка
    analytical_engine = AnalyticalEngine()
    result = analytical_engine.calculate(call_option, market_data)
    
    print(f"Европейский опцион колл:")
    print(f"  Цена: ${result['price']:.4f}")
    print(f"  Дельта: {result['delta']:.4f}")
    print(f"  Гамма: {result['gamma']:.6f}")
    print(f"  Тета: {result['theta']:.4f} в день")
    print(f"  Вега: {result['vega']:.4f} за 1% изменение волатильности")
    print(f"  Ро: {result['rho']:.4f} за 1% изменение ставки")
    print()

def implied_volatility_example():
    """Пример расчета подразумеваемой волатильности."""
    print("=== Расчет подразумеваемой волатильности ===")
    
    # Рыночные данные
    market_data = MarketData(
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.01
    )
    
    # Создаем опцион
    option = EuropeanOption(
        strike_price=100.0,
        expiration_time=1.0,
        option_type=OptionType.CALL
    )
    option.underlying_price = 100.0
    
    # Сначала рассчитываем справедливую цену
    analytical_engine = AnalyticalEngine()
    fair_price = analytical_engine.calculate(option, market_data)['price']
    
    # Рассчитываем подразумеваемую волатильность из рыночной цены
    market_price = fair_price
    implied_vol = calculate_implied_volatility(option, market_price, market_data)
    
    print(f"Опцион с волатильностью 20% имеет справедливую цену: ${fair_price:.4f}")
    print(f"Если рыночная цена равна ${market_price:.4f}, подразумеваемая волатильность: {implied_vol:.4f}")
    print()

def american_option_example():
    """Пример оценки американских опционов."""
    print("=== Оценка американских опционов ===")
    
    # Рыночные данные
    market_data = MarketData(
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.01
    )
    
    # Создаем американский опцион колл
    american_option = AmericanOption(
        strike_price=100.0,
        expiration_time=1.0,
        option_type=OptionType.CALL
    )
    american_option.underlying_price = 100.0
    
    # Оценка с использованием биномиального дерева
    binomial_engine = BinomialEngine()
    result = binomial_engine.calculate(american_option, market_data, steps=1000)
    
    print(f"Цена американского опциона колл: ${result['price']:.4f}")
    
    # Сравниваем с европейским опционом
    european_option = EuropeanOption(
        strike_price=100.0,
        expiration_time=1.0,
        option_type=OptionType.CALL
    )
    european_option.underlying_price = 100.0
    
    analytical_engine = AnalyticalEngine()
    european_price = analytical_engine.calculate(european_option, market_data)['price']
    
    print(f"Цена европейского опциона колл: ${european_price:.4f}")
    print(f"Премия за досрочное исполнение: ${result['price'] - european_price:.4f}")
    print()

def risk_metrics_example():
    """Пример расчета метрик риска."""
    print("=== Расчет метрик риска ===")
    
    # Генерируем примерные доходности
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)
    
    # Рассчитываем метрики риска
    var_95 = RiskMetrics.calculate_value_at_risk(returns, 0.95)
    es_95 = RiskMetrics.calculate_expected_shortfall(returns, 0.95)
    sharpe = RiskMetrics.calculate_sharpe_ratio(returns, 0.05/252)
    max_drawdown = RiskMetrics.calculate_max_drawdown(returns)
    
    print(f"Value at Risk (95%): {var_95:.6f}")
    print(f"Expected Shortfall (95%): {es_95:.6f}")
    print(f"Коэффициент Шарпа: {sharpe:.6f}")
    print(f"Максимальная просадка: {max_drawdown:.6f}")
    print()

if __name__ == "__main__":
    print("Примеры использования библиотеки BSM_pet - Русская версия")
    print("=" * 60)
    print()
    
    basic_option_pricing()
    implied_volatility_example()
    american_option_example()
    risk_metrics_example()
    
    print("Все примеры успешно завершены!")
