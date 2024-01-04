import numpy as np


class RegressionTreeNode:
    def __init__(
        self,
        value=None,
        left=None,
        right=None,
        feature=None,
        threshold=None,
        loss=None,
    ):
        # Structure
        self.value = value
        self.left = left
        self.right = right

        # The split feature and threshold
        self.feature = feature
        self.threshold = threshold

        # The loss of the split
        self.loss = loss


class RegressionTree:
    def __init__(self, max_depth=5, min_samples_split=1):
        self.root = RegressionTreeNode()
        self.depth = 1
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.root.value = y.mean()

        # BFS
        queue = [(self.depth + 1, self.root, X, y)]
        while queue:
            depth, node, _X, _y = queue.pop(0)

            # If the depth of the tree has exceeded the max depth, return the tree
            if depth > self.max_depth:
                break

            # If there are not enough samples to split, stop
            # If all the samples are the same, stop
            if len(_y) < self.min_samples_split or all(_y == _y[0]):
                continue

            # Traverse all the features
            selected_feature = None
            selected_threshold = None
            loss_buffer = float("inf")
            for feature_idx in range(_X.shape[1]):
                col = _X[:, feature_idx]

                # Remove duplicated values in the feature(column)
                deduplicated_value = set(col)
                if len(deduplicated_value) == 1:
                    continue

                # Remove the min value to avoid the case that all the values are the same
                deduplicated_value.remove(min(deduplicated_value))

                # Traverse all the possible threshold to get the split
                for threshold in deduplicated_value:
                    y_left = _y[col < threshold]
                    y_right = _y[col >= threshold]

                    loss = (
                        ((y_left - y_left.mean()) ** 2).sum()
                        + ((y_right - y_right.mean()) ** 2).sum()
                    ) / len(_y)

                    if loss_buffer > loss:
                        selected_threshold = threshold
                        selected_feature = feature_idx
                        loss_buffer = loss

            if selected_threshold is None:
                continue

            # Update the tree structure
            node.feature = selected_feature
            node.threshold = selected_threshold
            idx_left = _X[:, node.feature] < node.threshold
            idx_right = _X[:, node.feature] >= node.threshold
            node.left = RegressionTreeNode(value=_y[idx_left].mean())
            node.right = RegressionTreeNode(value=_y[idx_right].mean())
            queue.append((depth + 1, node.left, _X[idx_left], _y[idx_left]))
            queue.append((depth + 1, node.right, _X[idx_right], _y[idx_right]))

        self.depth = depth

    def predict_one(self, row):
        node = self.root
        while node.left and node.right:
            if row[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right

        return node.value

    def predict(self, X):
        return np.apply_along_axis(self.predict_one, 1, X)


if __name__ == "__main__":
    pass
