import pandas as pd
import math

data = pd.read_csv("SDLC-models.csv", error_bad_lines=False)

class Question:
    
    def __init__(self, df, col, value):
        self.df = df
        self.col = col
        self.value = value

    def match(self, example):
        val = self.df[self.col][example]
        return val == self.value

    def __repr__(self):
        return "%s %s?" % (self.col, str(self.value))


def unique_vals(df, col):
    return df[col].unique().tolist()



def class_counts(df):
    """Counts the number of each type of example in a dataset."""
    counts = {}
    for row in df["Model"]:
        label = row
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def partition(df, col, val):
    """Partitions a dataset.
    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    question = Question(df, col, val)
    true_rows, false_rows = [], []
    
    for row in range(len(df[col])):
        if question.match(row):
            true_rows.append(df.loc[[row]])
        else:
            false_rows.append(df.loc[[row]])
    try:
        true_rows = pd.concat(true_rows).reset_index(drop=True)
    except ValueError:
        true_rows = []
    try:
        false_rows = pd.concat(false_rows).reset_index(drop=True)
    except ValueError:
        false_rows = []
    return true_rows, false_rows

def gini(df):
    """Calculate the Gini Impurity for a list of rows.
    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(df)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(df["Model"]))
        impurity -= prob_of_lbl**2
    return impurity

def info_gain(left, right, current_uncertainty):
    """Information Gain.
    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

def find_best_split(df):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  
    best_question = None
    current_uncertainty = gini(df)
    n_features = len(df.columns) - 1 
    
    for col in df:
        if col == "Model":
            continue
        values = unique_vals(df, col)

        for val in values: 
            
            question = Question(df, col, val)

            true_rows, false_rows = partition(df, col, val)

      
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            gain = info_gain(true_rows, false_rows, current_uncertainty)


            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question

class Leaf:
    """A Leaf node classifies data.
    This holds a dictionary of class (e.g., "V-shaped") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, df):
        self.predictions = class_counts(df)


class Decision_Node:
    """A Decision Node asks a question.
    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(df):
    """Builds the tree.
    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

 
    gain, question = find_best_split(df)

    if gain == 0:
        return Leaf(df)

    col, val = question.col, question.value
    true_rows, false_rows = partition(df, col, val)

    true_branch = build_tree(true_rows)

    false_branch = build_tree(false_rows)

    return Decision_Node(question, true_branch, false_branch)

def print_tree(node, spacing=""):
    
    """World's most elegant tree printing function."""

    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    print (spacing + str(node.question))

    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")



tree = build_tree(data)
print_tree(tree)


