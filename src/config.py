from dataclasses import dataclass

@dataclass
class StrategyConfig: # you can change this
    w_trend: float = 0.50
    w_momentum: float = 0.15
    w_volume: float = 0.15
    w_volatility: float = 0.10
    w_breadth: float = 0.10
    beta: float = 1.0 # estimated 
    enter_threshold: float = 0.6
    exit_threshold: float = 0.45
    hold_on_weakness: bool = False # optional 
