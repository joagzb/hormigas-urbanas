import logging
import numpy as np

from src.scripts.utils.roulette_selection import roulette_wheel_selection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_roulette_selection_picks_valid_index():
    """Picks an index within range with non-zero probability."""
    probabilities = [0.5, 0.25, 0.25]
    assert sum(probabilities) == 1.0

    winner_index_1_based = roulette_wheel_selection(probabilities)
    assert 1 <= winner_index_1_based <= len(probabilities)
    assert probabilities[winner_index_1_based - 1] > 0
    
    
def test_roulette_selection_zero_prob_never_selected():
    """Zero-probability classes are never selected under repeated draws."""
    np.random.seed(0)
    probabilities = [0.0, 1.0, 0.0]
    for _ in range(200):
        winner_index_1_based = roulette_wheel_selection(probabilities)
        assert winner_index_1_based == 2
      
