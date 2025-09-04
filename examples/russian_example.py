"""
BSM_pet библиотека

Этот пример демонстрирует все функции библиотеки BSM_pet
включая оценку опционов, расчет греков, подразумеваемую волатильность,
метрики риска и различные типы опционов.
"""

import numpy as np
from bsm_pet import (
    MarketData, EuropeanOption, AmericanOption, BermudanOption, AsianOption,
    OptionType, ExerciseType, AnalyticalEngine, BinomialEngine,
    MonteCarloEngine, LongstaffSchwartzEngine,
    calculate_implied_volatility, RiskMetrics
)

def demonstrate_all_option_types():
    """Демонстрация оценки всех типов опционов."""
    print("=== Оценка всех типов опционов ===")
    
    # Рыночные данные
    market_data = MarketData(
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.01
    )
    
    # Создаем различные типы опционов с одинаковыми параметрами
    S = 100.0
    K = 100.0
    T = 1.0
    
    # Европейский опцион
    european_call = EuropeanOption(K, T, OptionType.CALL)
    european_call.underlying_price = S
    
    # Американский опцион
    american_call = AmericanOption(K, T, OptionType.CALL)
    american_call.underlying_price = S
    
    # Бермудский опцион (исполнение поквартально)
    bermudan_call = BermudanOption(K, T, OptionType.CALL, [0.25, 0.5, 0.75, 1.0])
    bermudan_call.underlying_price = S
    
    # Азиатский опцион (усреднение по определенным датам)
    asian_call = AsianOption(K, T, OptionType.CALL, [0.5, 0.75, 1.0])
    asian_call.underlying_price = S
    
    # Оценка с использованием соответствующих движков
    analytical_engine = AnalyticalEngine()
    binomial_engine = BinomialEngine()
    mc_engine = MonteCarloEngine(seed=42)
    ls_engine = LongstaffSchwartzEngine()
    
    # Европейский опцион (аналитический)
    european_result = analytical_engine.calculate(european_call, market_data)
    print(f"Европейский колл: ${european_result['price']:.4f}")
    
    # Американский опцион (биномиальный)
    american_result = binomial_engine.calculate(american_call, market_data, steps=1000)
    print(f"Американский колл: ${american_result['price']:.4f}")
    
    # Бермудский опцион (биномиальный)
    bermudan_result = binomial_engine.calculate(bermudan_call, market_data, steps=1000)
    print(f"Бермудский колл: ${bermudan_result['price']:.4f}")
    
    # Азиатский опцион (Монте-Карло)
    asian_result = mc_engine.calculate(asian_call, market_data, num_simulations=50000)
    print(f"Азиатский колл: ${asian_result['price']:.4f} ± {asian_result['standard_error']:.4f}")
    
    # Американский опцион (Лонгстаффа-Шварца)
    american_ls_result = ls_engine.calculate(american_call, market_data, num_simulations=10000)
    print(f"Американский колл (LS): ${american_ls_result['price']:.4f}")
    
    print()

def demonstrate_all_greeks():
    """Демонстрация расчета всех греков."""
    print("=== Расчет всех греков ===")
    
    # Рыночные данные
    market_data = MarketData(
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.01
    )
    
    # Создаем европейский опцион колл
    option = EuropeanOption(
        strike_price=100.0,
        expiration_time=1.0,
        option_type=OptionType.CALL
    )
    option.underlying_price = 100.0
    
    # Рассчитываем все греки с помощью аналитического движка
    analytical_engine = AnalyticalEngine()
    result = analytical_engine.calculate(option, market_data)
    
    print("Греки европейского опциона колл:")
    print(f"Цена: ${result['price']:.4f}")
    print(f"Дельта: {result['delta']:.4f}")
    print(f"Гамма: {result['gamma']:.6f}")
    print(f"Тета: {result['theta']:.4f} в день")
    print(f"Вега: {result['vega']:.4f} за 1% изменение волатильности")
    print(f"Ро: {result['rho']:.4f} за 1% изменение ставки")
    
    # Рассчитываем численные греки для сравнения
    binomial_engine = BinomialEngine()
    numerical_greeks = binomial_engine.calculate_greeks(option, market_data)
    
    print("\nЧисленные греки (Биномиальный):")
    print(f"Дельта: {numerical_greeks['delta']:.4f}")
    print(f"Гамма: {numerical_greeks['gamma']:.6f}")
    print(f"Тета: {numerical_greeks['theta']:.4f} в день")
    print(f"Вега: {numerical_greeks['vega']:.4f} за 1% изменение волатильности")
    print(f"Ро: {numerical_greeks['rho']:.4f} за 1% изменение ставки")
    
    print()

def demonstrate_implied_volatility():
    """Демонстрация расчета подразумеваемой волатильности."""
    print("=== Расчет подразумеваемой волатильности ===")
    
    # Рыночные данные
    market_data = MarketData(
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.01
    )
    
    # Создаем европейский опцион колл
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
    
    # Тестируем с разными рыночными ценами
    for premium in [0.5, 1.0, 2.0]:
        test_price = fair_price + premium
        test_implied_vol = calculate_implied_volatility(option, test_price, market_data)
        print(f"Рыночная цена ${test_price:.4f} → Подразумеваемая волатильность: {test_implied_vol:.4f}")
    
    print()

def demonstrate_risk_metrics():
    """Демонстрация расчета всех метрик риска."""
    print("=== Расчет всех метрик риска ===")
    
    # Генерируем примерные доходности
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)
    
    # Рассчитываем все метрики риска
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
    print(f"Коэффициент Шарпа: {sharpe:.6f}")
    print(f"Коэффициент Сортино: {sortino:.6f}")
    print(f"Максимальная просадка: {max_dd:.6f}")
    
    # Интерпретация
    print("\nИнтерпретация:")
    print(f"• Коэффициент Сортино ({sortino:.3f}) выше коэффициента Шарпа ({sharpe:.3f})")
    print("  что указывает на большую позитивную волатильность по сравнению с негативной")
    print(f"• 95% VaR {var_95:.4f} означает, что мы ожидаем потери больше этой")
    print("  величины только в 5% случаев")
    print(f"• Максимальная просадка {max_dd:.2%} показывает наибольшее падение от пика до дна")
    
    print()

def demonstrate_put_options():
    """Демонстрация оценки опционов пут."""
    print("=== Оценка опционов пут ===")
    
    # Рыночные данные
    market_data = MarketData(
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.01
    )
    
    # Создаем европейский опцион пут
    put_option = EuropeanOption(
        strike_price=100.0,
        expiration_time=1.0,
        option_type=OptionType.PUT
    )
    put_option.underlying_price = 100.0
    
    # Рассчитываем цену и греки
    analytical_engine = AnalyticalEngine()
    result = analytical_engine.calculate(put_option, market_data)
    
    print("Европейский опцион пут:")
    print(f"Цена: ${result['price']:.4f}")
    print(f"Дельта: {result['delta']:.4f}")
    print(f"Гамма: {result['gamma']:.6f}")
    print(f"Тета: {result['theta']:.4f} в день")
    print(f"Вега: {result['vega']:.4f} за 1% изменение волатильности")
    print(f"Ро: {result['rho']:.4f} за 1% изменение ставки")
    
    # Сравниваем с опционом колл
    call_option = EuropeanOption(
        strike_price=100.0,
        expiration_time=1.0,
        option_type=OptionType.CALL
    )
    call_option.underlying_price = 100.0
    
    call_result = analytical_engine.calculate(call_option, market_data)
    
    # Проверяем паритет пут-колл
    put_call_parity = result['price'] - call_result['price'] + 100.0 - 100.0 * np.exp(-0.05 * 1.0)
    print(f"\nПроверка паритета пут-колл: {put_call_parity:.6f} (должно быть близко к 0)")
    
    print()

def demonstrate_different_strikes():
    """Демонстрация оценки опционов с разными страйками."""
    print("=== Разные страйк-цены ===")
    
    # Рыночные данные
    market_data = MarketData(
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.01
    )
    
    # Создаем опционы с разными страйками
    strikes = [90.0, 100.0, 110.0]
    expiration_time = 1.0
    underlying_price = 100.0
    
    analytical_engine = AnalyticalEngine()
    
    print("Цены опционов колл с разными страйками:")
    for strike in strikes:
        option = EuropeanOption(strike, expiration_time, OptionType.CALL)
        option.underlying_price = underlying_price
        result = analytical_engine.calculate(option, market_data)
        moneyness = "ITM" if strike < underlying_price else "ATM" if strike == underlying_price else "OTM"
        print(f"K={strike:.1f} ({moneyness}): ${result['price']:.4f}")
    
    print("\nЦены опционов пут с разными страйками:")
    for strike in strikes:
        option = EuropeanOption(strike, expiration_time, OptionType.PUT)
        option.underlying_price = underlying_price
        result = analytical_engine.calculate(option, market_data)
        moneyness = "ITM" if strike > underlying_price else "ATM" if strike == underlying_price else "OTM"
        print(f"K={strike:.1f} ({moneyness}): ${result['price']:.4f}")
    
    print()

def demonstrate_time_decay():
    """Демонстрация временного распада (тета)."""
    print("=== Временной распад (Тета) ===")
    
    # Рыночные данные
    market_data = MarketData(
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.01
    )
    
    # Создаем опционы с разным временем до экспирации
    times = [1.0, 0.5, 0.25, 0.1]
    strike_price = 100.0
    underlying_price = 100.0
    
    analytical_engine = AnalyticalEngine()
    
    print("Тета опционов колл для разного времени до экспирации:")
    for time in times:
        option = EuropeanOption(strike_price, time, OptionType.CALL)
        option.underlying_price = underlying_price
        result = analytical_engine.calculate(option, market_data)
        print(f"T={time:.2f} лет: Тета = {result['theta']:.4f} в день")
    
    print("\nТета опционов пут для разного времени до экспирации:")
    for time in times:
        option = EuropeanOption(strike_price, time, OptionType.PUT)
        option.underlying_price = underlying_price
        result = analytical_engine.calculate(option, market_data)
        print(f"T={time:.2f} лет: Тета = {result['theta']:.4f} в день")
    
    print()

if __name__ == "__main__":
    print("BSM_pet библиотека - Полный пример использования (Русский)")
    print("=" * 60)
    print()
    
    demonstrate_all_option_types()
    demonstrate_all_greeks()
    demonstrate_implied_volatility()
    demonstrate_risk_metrics()
    demonstrate_put_options()
    demonstrate_different_strikes()
    demonstrate_time_decay()
    
    print("Все примеры успешно завершены!")
