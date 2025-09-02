import logging
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from scripts.utils.roulette_selection import (
    roulette_wheel_selection
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_roulette_selection():
    classes = [0.5, 0.25, 0.25]
    
    assert sum(classes) == 1.0
    
    logger.info(
        "Testing test_roulette_selection with 3 classes with probabilities 50%, 25%, 25% repectively",
    )
    winner = roulette_wheel_selection(classes)
    logger.info("selected %s class as winner", winner - 1)
    assert 1 <= winner <= len(classes)
    assert classes[winner - 1] > 0
    
    
def test_roulette_selection_zero_prob_never_selected():
    np.random.seed(0)
    classes = [0.0, 1.0, 0.0]
    for _ in range(200):
        winner = roulette_wheel_selection(classes)
        assert winner == 2
      