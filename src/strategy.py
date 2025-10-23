import pandas as pd
from .features import trend_features, momentum_features, volume_features, volatility_features, breadth_features
from .signals import combine_scores, to_signal
from .position import position_size
from .config import StrategyConfig

def build_strategy_signals(tdf: pd.DataFrame,
                           market_df: pd.DataFrame,
                           sector_df: pd.DataFrame,
                           vix_df: pd.DataFrame,
                           cfg: StrategyConfig) -> pd.DataFrame:
    # 1) features
    trend_s   = trend_features(tdf, ref_df=sector_df)
    mom_s     = momentum_features(tdf)
    vol_s     = volume_features(tdf)
    vola_s    = volatility_features(tdf)
    breadth_s = breadth_features(market_df, vix_df, beta=cfg.beta)
    # 2) combine
    combo     = combine_scores(cfg, trend_s, mom_s, vol_s, vola_s, breadth_s)
    signal    = to_signal(cfg, combo)
    size      = position_size(tdf) * signal  # zero when flat
    out = pd.DataFrame({
        'TrendS': trend_s,
        'MomS': mom_s,
        'VolS': vol_s,
        'VolaS': vola_s,
        'BreadthS': breadth_s,
        'Score': combo,
        'Signal': signal,
        'PosSize': size
    })
    return out
