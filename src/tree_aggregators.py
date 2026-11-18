from typing import Dict, List, Tuple

import numpy as np

from classification_tree_builder import GenerationTree, TreeNode


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    return shifted - np.log(np.exp(shifted).sum())


class TreeAggregator:
    """Aggregates the per-token log probabilities stored on a GenerationTree
    into a probability distribution over the class labels.

    One instance wraps a single tree (i.e. a single data point). Each public
    aggregation method returns a dict mapping label -> probability, already
    normalised across labels so the methods can be compared directly.

    Properties:
        tree (GenerationTree): the tree being aggregated.
        class_labels (List[str]): the labels the tree was built for.
        label_paths (Dict[str, List[TreeNode]]): the path of nodes, from the
            first token to the <end> node, for each label.
    """

    def __init__(self, tree: GenerationTree):
        self.tree = tree
        self.class_labels = tree.class_labels
        self.label_paths: Dict[str, List[TreeNode]] = self._extract_label_paths()

    def _extract_label_paths(self) -> Dict[str, List[TreeNode]]:
        """Walk the trie and collect the node path for every label.

        Returns:
            Dict[str, List[TreeNode]]: label -> list of nodes from the first
                token up to and including the <end> (-1) node.
        """
        paths: Dict[str, List[TreeNode]] = {}

        def walk(node: TreeNode, path: List[TreeNode]):
            if not node.children:
                # a leaf is always an <end> node and marks the end of a label
                paths[node.label_name] = path
                return
            for child_node in node.children.values():
                walk(child_node, path + [child_node])

        walk(self.tree.root, [])
        return paths

    def _path_log_probs(self, path: List[TreeNode]) -> np.ndarray:
        return np.array([node.log_prob for node in path], dtype=np.float64)

    def _path_probs(self, path: List[TreeNode]) -> np.ndarray:
        return np.array([node.prob for node in path], dtype=np.float64)

    def _decision_path_log_probs(self, path: List[TreeNode]) -> np.ndarray:
        return np.array(
            [
                node.log_prob
                for i, node in enumerate(path)
                if (i == 0 or path[i - 1].has_multiple_children())
                # first node always there (not root), then dependent on
                # how many children parent has
            ]
        )

    def _divergence_index(self, path: List[TreeNode]) -> int:
        """Index of the first divergent node in a path.

        The divergent suffix (unique to this label) starts here. The <end>
        node is always divergent, so an index is always found.
        """
        for i, node in enumerate(path):
            if node.is_divergent:
                return i
        return len(path) - 1

    def _softmax_labels(self, scores: Dict[str, float]) -> Dict[str, float]:
        labels = list(scores.keys())
        probs = _softmax(np.array([scores[label] for label in labels]))
        return {label: float(prob) for label, prob in zip(labels, probs)}

    def _normalize_labels(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalise non-negative label scores (probabilities) to sum to 1."""
        total = sum(scores.values())
        if total <= 0:
            raise ValueError(
                f"Label scores sum to {total}; expected positive probabilities."
            )
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Probability-tree scores sum to {total}, expected ~1.0.")
        return {label: score / total for label, score in scores.items()}

    def greedy(self) -> Dict[str, float]:
        """Greedy decode through the constrained tree: from the root, follow the
        highest-log-prob child at each step until reaching an <end> leaf. The
        label at that leaf gets 1.0, every other label 0.0.

        This emulates greedy decoding masked to the label set — the label the
        model would emit token-by-token if it could only produce valid label
        tokens — rather than scoring whole sequences.
        """
        node = self.tree.root
        while node.children:
            node = max(node.children.values(), key=lambda child: child.log_prob)
        winner = node.label_name
        return {label: float(label == winner) for label in self.label_paths}

    def full_log_prob(self, decision_only=False) -> Dict[str, float]:
        """Sum every node log prob along the path, then softmax over labels."""
        if decision_only:
            path_log_probs = self._decision_path_log_probs
        else:
            path_log_probs = self._path_log_probs
        scores = {
            label: float(path_log_probs(path).sum())
            for label, path in self.label_paths.items()
        }
        return self._softmax_labels(scores)

    def full_log_prob_decision(self) -> Dict[str, float]:
        return self.full_log_prob(decision_only=True)

    def length_normalized(self, decision_only=False) -> Dict[str, float]:
        """Length normalisation: sum of log probs divided by the number of
        tokens, then softmax over labels."""
        if decision_only:
            path_log_probs = self._decision_path_log_probs
        else:
            path_log_probs = self._path_log_probs
        scores = {}
        for label, path in self.label_paths.items():
            log_probs = path_log_probs(path)
            scores[label] = float(log_probs.sum() / len(log_probs))
        return self._softmax_labels(scores)

    def length_normalized_decision(self):
        return self.length_normalized(decision_only=True)

    def google_length_normalized(
        self, decision_only: bool = False, alpha: float = 0.6
    ) -> Dict[str, float]:
        """GNMT length penalty: sum of log probs divided by
        ((5 + L) ** alpha) / ((5 + 1) ** alpha), then softmax over labels.

        Args:
            alpha (float): length penalty strength. 0 recovers the raw sum,
                larger values favour longer labels more strongly.
        """
        if decision_only:
            path_log_probs = self._decision_path_log_probs
        else:
            path_log_probs = self._path_log_probs
        scores = {}
        for label, path in self.label_paths.items():
            log_probs = path_log_probs(path)
            length = len(log_probs)
            penalty = ((5 + length) ** alpha) / ((5 + 1) ** alpha)
            scores[label] = float(log_probs.sum() / penalty)
        return self._softmax_labels(scores)

    def google_length_normalized_decision(self, alpha: float = 0.6):
        return self.google_length_normalized(alpha=alpha, decision_only=True)

    def decision_only(self) -> Dict[str, float]:
        # only get the decision nodes
        return {"x": 21.1}

    # def divergence_normalised(self) -> Dict[str, float]:
    # """Only sum the log probs of the divergent suffix (the tokens unique
    # to the label), length normalise, then softmax over labels."""
    # scores = {}
    # for label, path in self.label_paths.items():
    # start = self._divergence_index(path)
    # log_probs = self._path_log_probs(path)[start:]
    # scores[label] = float(log_probs.sum() / len(log_probs))
    # return self._softmax_labels(scores)

    # def weighted_divergence(self, decay: float = 0.5) -> Dict[str, float]:
    # """Weighted average of the divergent suffix log probs, with
    # geometrically decaying weights so the first (most discriminative)
    # token dominates, then softmax over labels.

    # Args:
    # decay (float): geometric decay factor in (0, 1]. Smaller values
    # concentrate more weight on the first divergent token; 1.0
    # recovers a plain average (i.e. divergence_normalised).
    # """
    # scores = {}
    # for label, path in self.label_paths.items():
    # start = self._divergence_index(path)
    # log_probs = self._path_log_probs(path)[start:]
    # weights = decay ** np.arange(len(log_probs))
    # weights /= weights.sum()
    # scores[label] = float((weights * log_probs).sum())
    # return self._softmax_labels(scores)

    def probability_tree(self) -> Dict[str, float]:
        """Sum every node log prob along the path, then softmax over labels."""
        scores = {
            label: float(self._path_probs(path).prod())
            for label, path in self.label_paths.items()
        }
        return self._normalize_labels(scores)

    def predict(self, method: str, **kwargs) -> Tuple[str, float, Dict[str, float]]:
        """Run one aggregation method and return the argmax label.

        Args:
            method (str): a key of ``AGGREGATION_METHODS``.

        Returns:
            Tuple[str, float, Dict[str, float]]: predicted label, its
                probability, and the soft label -> probability distribution.
        """
        probs = AGGREGATION_METHODS[method](self, **kwargs)
        pred = max(probs, key=probs.get)
        return pred, probs[pred], probs


AGGREGATION_METHODS = {
    "full_log_prob": TreeAggregator.full_log_prob,
    "length_norm": TreeAggregator.length_normalized,
    "google_length_norm": TreeAggregator.google_length_normalized,
    "full_log_prob_decision": TreeAggregator.full_log_prob_decision,
    "length_norm_decision": TreeAggregator.length_normalized_decision,
    "google_length_norm_decision": TreeAggregator.google_length_normalized_decision,
    "probability_tree": TreeAggregator.probability_tree,
    "greedy": TreeAggregator.greedy,
}
