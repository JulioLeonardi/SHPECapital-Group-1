# signals.py
import numpy as np
import pandas as pd

def dynamic_weights(cfg, breadth_series: pd.Series) -> dict:
    # example: reduce breadth weight if beta < 0.8
    b_adj = cfg.w_breadth * (0.5 if cfg.beta < 0.8 else 1.0)
    total = cfg.w_trend + cfg.w_momentum + cfg.w_volume + cfg.w_volatility + b_adj
    # rescale to sum to 1
    return {
        'trend': cfg.w_trend / total,
        'momentum': cfg.w_momentum / total,
        'volume': cfg.w_volume / total,
        'volatility': cfg.w_volatility / total,
        'breadth': b_adj / total,
    }

def combine_scores(cfg, trend_s, mom_s, vol_s, vola_s, breadth_s) -> pd.Series:
    w = dynamic_weights(cfg, breadth_s)
    combo = (w['trend']*trend_s + w['momentum']*mom_s + w['volume']*vol_s +
             w['volatility']*vola_s + w['breadth']*breadth_s)
    return combo.clip(0,1)

def to_signal(cfg, combo_score: pd.Series) -> pd.Series:
    # classic hysteresis entry/exit
    state = 0
    sig = []
    for x in combo_score.fillna(0):
        if state == 0 and x >= cfg.enter_threshold:
            state = 1
        elif state == 1 and x <= cfg.exit_threshold:
            state = 0
        sig.append(state)
    return pd.Series(sig, index=combo_score.index, name='Signal')
