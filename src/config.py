from dataclasses import dataclass

@dataclass
class StrategyConfig:
    w_trend: float = 0.30
    w_momentum: float = 0.20
    w_volume: float = 0.20
    w_volatility: float = 0.15
    w_breadth: float = 0.15
    beta: float = 1.0
    enter_threshold: float = 0.6
    exit_threshold: float = 0.45
    hold_on_weakness: bool = True
