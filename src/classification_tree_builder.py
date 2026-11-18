import copy
import json
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from graphviz import Source
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class TreeNode:
    """Class to represent a tree/trie node, where each tree node represents a
        token in the sequence.

    Properties:
        children (Dict[int, TreeNode]): children of the node.
        depth (int): the depth of the node within the tree.
        label_name (Optional[str]): set only when is the end of a label.
        is_end_of_label (bool): is this token the end of a label?
    """

    def __init__(self, value, depth=0):
        self.value = value
        self.children: Dict[int, TreeNode] = {}
        self.depth: int = depth

        # to set during tree building
        self.label_name: Optional[str] = None
        self.is_end_of_label: bool = False
        self.is_divergent: bool = False

        self.log_prob = 0.0

        # additional value placeholder for method that involve
        # softmaxxing over node siblings
        self.prob = -1.0

    def has_children(self) -> bool:
        return len(self.children) > 0

    def has_multiple_children(self) -> bool:
        return len(self.children) > 1

    def has_one_child(self) -> bool:
        return len(self.children) == 1

    def __str__(self):
        return str(self.value)


class GenerationTree:

    def __init__(self, class_labels, tokenizer, single_token_classes=False):
        self.class_labels = class_labels
        self.tokenizer = tokenizer
        self.single_token_classes = single_token_classes

        self.root = TreeNode(None, depth=0)
        self.max_label_len = 0

        self._build_tree()

    def _build_tree(self):
        for label in self.class_labels:
            # TODO: add label string manipulation
            token_ids = self.tokenizer.encode(label, add_special_tokens=False)
            # option to not add end of generation token (saves decode time for single token mode)
            if self.single_token_classes:
                if len(token_ids) != 1:
                    raise ValueError(
                        "Single token classes set to True, but label is not a single token."
                    )
            else:
                token_ids.append(-1)
            self.max_label_len = max(self.max_label_len, len(token_ids))

            # insert label into trie
            node = self.root
            for tid in token_ids:
                if tid not in node.children:
                    node.children[tid] = TreeNode(tid, depth=int(node.depth + 1))
                node = node.children[tid]
            node.is_end_of_label = True
            node.label_name = label
        self._set_divergent_nodes(node=self.root)

    def _set_divergent_nodes(self, node: Optional[TreeNode] = None):
        if node is None:
            node = self.root

        # because it is a leaf, it must be end of label
        if not node.has_children():
            node.is_divergent = True
            return node

        if node.has_one_child():
            # TODO: tidy this up
            child_node = self._set_divergent_nodes(node=list(node.children.values())[0])
            node.is_divergent = child_node.is_divergent
            node.label_name = child_node.label_name
            return node

        node.is_divergent = False
        node.label_name = None
        for child_node in node.children.values():
            self._set_divergent_nodes(child_node)
        return node

    def _node_to_dict(self, node: TreeNode):
        return {
            "value": node.value,
            "depth": node.depth,
            "label_name": node.label_name,
            "is_end_of_label": node.is_end_of_label,
            "log_prob": node.log_prob,
            "children": {
                str(tok): self._node_to_dict(child_node)
                for tok, child_node in node.children.items()
            },
        }

    def set_sibling_probs(self, node: Optional[TreeNode] = None):
        """Softmax each node's log prob over its siblings and store the result
        in node.prob, so the probability-tree aggregation can read it off
        directly rather than recomputing it on the fly.

        Args:
            node (Optional[TreeNode]): subtree root to process. Defaults to the
                tree root.
        """
        if node is None:
            node = self.root

        if not node.has_children():
            return

        children = list(node.children.values())
        sibling_log_probs = torch.tensor([child.log_prob for child in children])
        sibling_probs = F.softmax(sibling_log_probs, dim=0)
        for child_node, prob in zip(children, sibling_probs):
            child_node.prob = prob.item()
            self.set_sibling_probs(child_node)

    def save(self, save_path):
        payload = {
            "class_labels": self.class_labels,
            "max_label_len": self.max_label_len,
            "root": self._node_to_dict(self.root),
        }
        with open(save_path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def _node_from_dict(cls, node_dict):
        node = TreeNode(node_dict["value"], depth=node_dict["depth"])
        node.label_name = node_dict["label_name"]
        node.is_end_of_label = node_dict["is_end_of_label"]
        node.log_prob = node_dict["log_prob"]
        node.children = {
            int(tok): cls._node_from_dict(child)
            for tok, child in node_dict["children"].items()
        }
        return node

    @classmethod
    def load(cls, load_path, tokenizer, get_divergence=False):
        with open(load_path, "r") as f:
            payload = json.load(f)

        # bypass init
        tree = cls.__new__(cls)
        tree.class_labels = payload["class_labels"]
        tree.max_label_len = payload["max_label_len"]
        tree.tokenizer = tokenizer
        tree.root = cls._node_from_dict(payload["root"])
        if get_divergence:
            tree._set_divergent_nodes(node=tree.root)
        tree.set_sibling_probs()
        return tree

    def node_text(self, node: TreeNode) -> str:
        """Get human-readable output for nodes.

        Args:
            node (TreeNode): the node to produce human-readable
                output for.

        Returns:
            str: human-readable output for that node.
        """
        if node.value is None:
            return "ROOT"
        if node.value == -1:
            return "⟨end⟩"
        if self.tokenizer is not None:
            return repr(self.tokenizer.decode([node.value]))
        return str(node.value)

    def to_dot(self, show_probs=None):
        """show_probs: None (no edge labels), 'log' (log_prob), or 'prob' (exp)."""
        lines = ["digraph trie {", "  rankdir=LR;", '  node [fontname="monospace"];']
        ids, counter = {}, [0]

        def nid(node):
            key = id(node)
            if key not in ids:
                ids[key] = f"n{counter[0]}"
                counter[0] += 1
            return ids[key]

        def esc(s):
            return s.replace("\\", "\\\\").replace('"', '\\"')

        def edge_label(child):
            if show_probs is None:
                return ""
            value = child.log_prob if show_probs == "log" else child.prob
            return f' [label="{value:.3f}", fontsize=10]'

        def visit(node):
            text = esc(self.node_text(node))
            shape = (
                "diamond"
                if node.value == -1
                else "box" if node.is_end_of_label else "ellipse"
            )
            attrs = [f'label="{text}"', f"shape={shape}"]
            if node.is_divergent:
                attrs += ["style=filled", 'fillcolor="#c8e6c9"']
            lines.append(f'  {nid(node)} [{", ".join(attrs)}];')
            for child in node.children.values():
                lines.append(f"  {nid(node)} -> {nid(child)}{edge_label(child)};")
                visit(child)

        visit(self.root)
        lines.append("}")
        return "\n".join(lines)

    def render(self, path="trie", fmt="png", view=False, show_probs=None):
        Source(self.to_dot(show_probs)).render(
            path, format=fmt, view=view, cleanup=True
        )

    def _describe(self, node):
        flags = []
        if node.is_end_of_label:
            flags.append(f"label={node.label_name!r}")
        if node.is_divergent:
            flags.append("divergent")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        return self.node_text(node) + suffix

    def print_tree(self):
        def walk(node, prefix, is_last):
            connector = "└── " if is_last else "├── "
            print(prefix + connector + self._describe(node))
            new_prefix = prefix + ("    " if is_last else "│   ")
            children = list(node.children.values())
            for i, c in enumerate(children):
                walk(c, new_prefix, i == len(children) - 1)

        print(self._describe(self.root))
        children = list(self.root.children.values())
        for i, c in enumerate(children):
            walk(c, "", i == len(children) - 1)


class TreeBuilder:
    """Builds generation trees with the log probs of each next token
    softmaxxed over the WHOLE vocabulary. This tree can then be used
    in the classification stage with various different ways to aggregate
    probabilities of each token that is part of a label.
    """

    def __init__(
        self,
        model_name: str,
        class_labels: List[str],
        device: str = "cuda",
        max_model_len=None,
        dtype=torch.float16,
        single_token_classes=False,
    ):
        # set model-related properties
        self.device = device
        self.max_model_len = max_model_len if max_model_len is not None else 2048
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
            trust_remote_code=True,
            dtype=dtype,
        )
        self.model.eval()
        self.classes = class_labels
        self.tree = GenerationTree(
            self.classes, self.tokenizer, single_token_classes=single_token_classes
        )

    def score_tree(self, full_prompt: str, eoa_tokens: Optional[List[int]] = None):
        if eoa_tokens is None:
            eoa_tokens = [self.tokenizer.eos_token_id]

        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            max_length=self.max_model_len,
            truncation=True,
        ).to(self.model.device)
        input_ids = inputs.input_ids

        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

        # calucalate probability for each label with recursive traversal
        # start with all children of root
        for child_node in self.tree.root.children.values():
            self._score_tree_recursively(
                child_node,
                next_token_logits,
                copy.deepcopy(past_key_values),
                eoa_tokens,
            )
        self.tree.set_sibling_probs()

    def _score_tree_recursively(
        self,
        node: TreeNode,
        current_logits: torch.Tensor,
        past_key_values: torch.Tensor,
        eoa_tokens: List[int],
    ):
        """
        DFS to calculate probability of every valid label in the Trie.
        Should not be passed root as None values will cause errors.
        """
        # normalise over whole vocab
        # P(token | context)
        log_probs = F.log_softmax(current_logits, dim=-1)

        # if is end of generation token
        if node.value == -1:
            node.log_prob = torch.logsumexp(log_probs[0, eoa_tokens], dim=0).item()
            return  # -1 always a leaf

        node.log_prob = log_probs[0, node.value].item()

        if not node.children:
            return

        # forward pass to get distribution for children
        next_input_id = torch.tensor([[node.value]], device=self.device)
        with torch.no_grad():
            outputs = self.model(
                next_input_id, past_key_values=past_key_values, use_cache=True
            )
            child_logits = outputs.logits[:, -1, :]
            child_kv = outputs.past_key_values

        for child_node in node.children.values():
            self._score_tree_recursively(
                child_node, child_logits, copy.deepcopy(child_kv), eoa_tokens
            )

        # memory cleanup
        del outputs, child_logits, child_kv, next_input_id
