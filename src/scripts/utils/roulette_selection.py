import numpy as np

def roulette_wheel_selection(classes_probabilities):
    """
    Performs roulette wheel selection based on given probabilities.

    Parameters:
    classes_probabilities: List or array of probabilities. e.g: [0.1, 0.2, 0.3, 0.4] means 4 classes, each position value representing the class probability of happen

    Returns:
    pos: The selected class based on the roulette wheel selection
    """
    classes_probabilities = np.array(classes_probabilities) / np.sum(classes_probabilities)
    winner = np.random.choice(len(classes_probabilities), p=classes_probabilities) + 1
    return winner