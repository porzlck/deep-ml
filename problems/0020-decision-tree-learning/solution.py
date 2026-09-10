import math
from collections import Counter


def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def calculate_information_gain(
    examples: list[dict], attr: str, target_attr: str
) -> float:
    """Calculate the information gain of splitting on attr."""
    labels = [example[target_attr] for example in examples]
    original_entropy = calculate_entropy(labels)
    groups = {}
    for example in examples:
        value = example[attr]
        groups.setdefault(value, []).append(example)
    weighted_entropy = 0.0
    total = len(examples)
    for group in groups.values():
        group_labels = [example[target_attr] for example in group]
        weighted_entropy += (len(group) / total) * calculate_entropy(group_labels)
    return original_entropy - weighted_entropy


def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    counts = Counter(example[target_attr] for example in examples)
    max_count = max(counts.values())
    candidates = sorted(label for label, count in counts.items() if count == max_count)
    return candidates[0]


def learn_decision_tree(
    examples: list[dict], attributes: list[str], target_attr: str
) -> dict:
    """Build a decision tree using the ID3 algorithm."""
    labels = [example[target_attr] for example in examples]
    if len(set(labels)) == 1:
        return labels[0]
    if not attributes:
        return majority_class(examples, target_attr)
    best_attr = attributes[0]
    best_gain = calculate_information_gain(examples, best_attr, target_attr)
    for attr in attributes[1:]:
        gain = calculate_information_gain(examples, attr, target_attr)
        if gain > best_gain:
            best_gain = gain
            best_attr = attr
    tree = {best_attr: {}}
    remaining_attributes = [attr for attr in attributes if attr != best_attr]
    values = sorted({example[best_attr] for example in examples})
    for value in values:
        subset = [example for example in examples if example[best_attr] == value]
        tree[best_attr][value] = learn_decision_tree(
            subset, remaining_attributes, target_attr
        )
    return tree
