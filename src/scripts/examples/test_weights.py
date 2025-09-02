import unittest

from ..utils.weights import get_connection_weight


class TestWeights(unittest.TestCase):
    def test_missing_edge_returns_none(self):
        graph = {
            "connections": {0: [1]},
            "weights": {0: [1.0]},
        }

        result = get_connection_weight(graph, 0, 2)
        self.assertIsNone(result)

    def test_missing_start_node_returns_none(self):
        graph = {
            "connections": {0: [1]},
            "weights": {0: [1.0]},
        }

        result = get_connection_weight(graph, 3, 1)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
