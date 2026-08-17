"""Historical replay, validation, and execution modelling.

The live agent learns one day at a time. This package lets it learn from years
in an afternoon — but only if the replay is honest, which is the entire design
concern here. Every component is built so that lookahead is structurally
impossible rather than merely discouraged.
"""

from .datastore import Bar, LookaheadError, PointInTimeStore  # noqa: F401
from .fills import FillModel, Fill  # noqa: F401
from .replay import BacktestEngine, BacktestResult  # noqa: F401
from .validation import (  # noqa: F401
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    purged_walk_forward_splits,
)
