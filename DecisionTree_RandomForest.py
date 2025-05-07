import numpy as np

"""
Decision Tree implementation using Iterative Dichotomiser 3 algorithm
Random Forest implementation make use of the TreeNode class (originally used for Decision Tree)
"""

class TreeNode(object):
    def __init__(self, ids = None, children = [], entropy = 0, depth = 0):
        self.ids = ids           # index of data in this node
        self.entropy = entropy   # entropy, will fill later
        self.depth = depth       # distance to root node
        self.split_attribute = None # which attribute is chosen, it non-leaf
        self.children = children # list of its child nodes
        self.order = None       # order of values of split_attribute in children
        self.label = None       # label of node if it is a leaf

    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute
        self.order = order

    def set_label(self, label):
        self.label = label


def entropy(freq):
    # remove prob 0 
    freq_0 = freq[np.array(freq).nonzero()[0]]
    prob_0 = freq_0/float(freq_0.sum())
    return -np.sum(prob_0*np.log(prob_0))

class DecisionTreeID3(object):
    def __init__(self, max_depth= 10, min_samples_split = 2, min_gain = 1e-4):
        self.root = None
        self.max_depth = max_depth 
        self.min_samples_split = min_samples_split 
        self.Ntrain = 0
        self.min_gain = min_gain
    
    def fit(self, data, target):
        self.Ntrain = data.count()[0]
        self.data = data 
        self.attributes = list(data)
        self.target = target 
        self.labels = target.unique()
        
        ids = range(self.Ntrain)
        self.root = TreeNode(ids = ids, entropy = self._entropy(ids), depth = 0)
        queue = [self.root]
        while queue:
            node = queue.pop()
            if node.depth < self.max_depth or node.entropy < self.min_gain:
                node.children = self._split(node)
                if not node.children: #leaf node
                    self._set_label(node)
                queue += node.children
            else:
                self._set_label(node)
                
    def _entropy(self, ids):
        # Calculate entropy for the node using the provided IDs.
        if len(ids) == 0:
            return 0
        freq = np.array(self.target.iloc[ids].value_counts())
        return entropy(freq)

    def _set_label(self, node):
        # Assign the most frequent label to a node.
        target_ids = node.ids
        node.set_label(self.target.iloc[target_ids].mode()[0])

    
    def _split(self, node):
        ids = node.ids 
        best_gain = 0
        best_splits = []
        best_attribute = None
        order = None
        sub_data = self.data.iloc[ids, :]
        for i, att in enumerate(self.attributes):
            values = self.data.iloc[ids, i].unique().tolist()
            if len(values) == 1: continue # entropy = 0
            splits = []
            for val in values: 
                sub_ids = sub_data.index[sub_data[att] == val].tolist()
                splits.append([sub_id-1 for sub_id in sub_ids])
            # don't split if a node has too small number of points
            if min(map(len, splits)) < self.min_samples_split: continue
            # information gain
            HxS= 0
            for split in splits:
                HxS += len(split)*self._entropy(split)/len(ids)
            gain = node.entropy - HxS 
            if gain < self.min_gain: continue # stop if small gain 
            if gain > best_gain:
                best_gain = gain 
                best_splits = splits
                best_attribute = att
                order = values
        node.set_properties(best_attribute, order)
        child_nodes = [TreeNode(ids = split,
                     entropy = self._entropy(split), depth = node.depth + 1) for split in best_splits]
        return child_nodes

    def predict(self, new_data):
        npoints = new_data.shape[0]
        labels = [None] * npoints

        for n in range(npoints):
            x = new_data.iloc[n, :]  # one point
            node = self.root

            while node.children:
                #print(f"Current Node Split Attribute: {node.split_attribute}")
                #print(f"Node Order: {node.order}")
                #print(f"Value in Test Data: {x[node.split_attribute]}")

                value = x[node.split_attribute]
                if value not in node.order:
                    #print(f"Value '{value}' not found in node order: {node.order}")
                    break
                node = node.children[node.order.index(value)]

            labels[n] = node.label if node.label is not None else None

        return labels
    
class RandomForest:
    def __init__(self, n_tree=3, max_depth=10, min_samples_split=2, min_gain=1e-4):
        self.n_tree = n_tree
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.trees = []

    def fit(self, X, Y):
        trees = []
        for i in range(self.n_tree):
            # Sample 70% of the data with replacement
            X_sub = X.sample(frac=0.7, replace=True, random_state=42 + i)
            indices = X_sub.index
            Y_sub = Y.loc[indices].reset_index(drop=True)
            X_sub = X_sub.reset_index(drop=True)
            
            # Initialize and fit a decision tree
            tree = DecisionTreeID3(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_gain=self.min_gain)
            tree.fit(X_sub, Y_sub)
            trees.append(tree)
        
        self.trees = trees

    def predict(self, X):
        # Aggregate predictions from all trees
        predictions = np.zeros((X.shape[0], self.n_tree))
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)
        
        # Replace NaN values with random 0s and 1s
        for i in range(predictions.shape[0]):
            for j in range(predictions.shape[1]):
                if np.isnan(predictions[i, j]):
                    predictions[i, j] = np.random.choice([0, 1])
        
        # Majority vote
        final_predictions = [np.bincount(predictions[i].astype(int)).argmax() for i in range(X.shape[0])]
        return final_predictions