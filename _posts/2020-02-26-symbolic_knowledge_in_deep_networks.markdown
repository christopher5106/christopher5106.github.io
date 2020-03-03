---
layout: post
title:  "Symbolic Knowledge in deep learning"
date:   2020-02-26 05:00:00
categories: deep learning
---

**Writing in progress. Delivery tonight**

In this post, I'll give some explanations about [Embedding Symbolic Knowledge into Deep Networks](https://arxiv.org/abs/1909.01161) paper and their [code](https://github.com/ZiweiXU/LENSR) for someone interested in the deep understanding of the implementation, not a global overview.

In particular, I'll answer the following questions:

1/ what are the features X of leaf nodes in the symbolic knowledge embedder ?

2/ How is computed the embedding of the formula ?

3/ How do we link formula satisfaction and model predictions during training ?

<span style="color:red">Un-answered questions or bugs are left in red.</span>

# The goal and datasets

The idea of the paper is to use Graph Convolutional Networks (GCN) to represent logical formulas with an embedding vector, and to use these representations as a new logical loss, either to predict if an assignment of variables will satisfy a logic, or to train a neural network's outputs to satisfy a logic.

The evaluation of the representations are performed on 2 datasets:

- a synthetic dataset of logical formulas, to predict if an assignment will satisfy the formulas

<span style="color:red">Q1: how are [logical formula strings](https://github.com/ZiweiXU/LENSR/blob/master/dataset/Synthetic/formula_strings_0606.pk) created ?</span>

<span style="color:red">Q2: where do [features of symbols and logic operands](https://github.com/ZiweiXU/LENSR/blob/master/model/pygcn/pygcn/features.pk) come from ?</span>

- the [Visual Relationship Detection (VRD)](https://cs.stanford.edu/people/ranjaykrishna/vrd/) dataset with its original images from [Scene Graph dataset](https://cs.stanford.edu/people/jcjohns/cvpr15_supp/).

The dataset is composed of 100 object classes (`objects.json`):

```
'person', 'sky', 'building', 'truck', 'bus', 'table', 'shirt', 'chair', 'car', 'train', 'glasses', 'tree', 'boat', 'hat', 'trees', 'grass', 'pants', 'road', 'motorcycle', 'jacket', 'monitor', 'wheel', 'umbrella', 'plate', 'bike', 'clock', 'bag', 'shoe', 'laptop', 'desk', 'cabinet', 'counter', 'bench', 'shoes', 'tower', 'bottle', 'helmet', 'stove', 'lamp', 'coat', 'bed', 'dog', 'mountain', 'horse', 'plane', 'roof', 'skateboard', 'traffic light', 'bush', 'phone', 'airplane', 'sofa', 'cup', 'sink', 'shelf', 'box', 'van', 'hand', 'shorts', 'post', 'jeans', 'cat', 'sunglasses', 'bowl', 'computer', 'pillow', 'pizza', 'basket', 'elephant', 'kite', 'sand', 'keyboard', 'plant', 'can', 'vase', 'refrigerator', 'cart', 'skis', 'pot', 'surfboard', 'paper', 'mouse', 'trash can', 'cone', 'camera', 'ball', 'bear', 'giraffe', 'tie', 'luggage', 'faucet', 'hydrant', 'snowboard', 'oven', 'engine', 'watch', 'face', 'street', 'ramp', 'suitcase'
```

70 predicates (`predicates.json`)

```
'on', 'wear', 'has', 'next to', 'sleep next to', 'sit next to', 'stand next to', 'park next', 'walk next to', 'above', 'behind', 'stand behind', 'sit behind', 'park behind', 'in the front of', 'under', 'stand under', 'sit under', 'near', 'walk to', 'walk', 'walk past', 'in', 'below', 'beside', 'walk beside', 'over', 'hold', 'by', 'beneath', 'with', 'on the top of', 'on the left of', 'on the right of', 'sit on', 'ride', 'carry', 'look', 'stand on', 'use', 'at', 'attach to', 'cover', 'touch', 'watch', 'against', 'inside', 'adjacent to', 'across', 'contain', 'drive', 'drive on', 'taller than', 'eat', 'park on', 'lying on', 'pull', 'talk', 'lean on', 'fly', 'face', 'play with', 'sleep on', 'outside of', 'rest on', 'follow', 'hit', 'feed', 'kick', 'skate on'
```

and the files `annotations_train.json` and `annotations_test.json`, dictionaries containing for each file a list of annotations inthe JSON format

```
{
  'predicate': 0,
  'object': {'category': 0, 'bbox': [318, 1019, 316, 767]},
  'subject': {'category': 13, 'bbox': [330, 473, 501, 724]}
}
```

From the dataset, logical formulas are computed to constrain each relation predicate with all possible relation position between objects. The goal here is to have a neural network prediction of a relation between objects to satisfy the constrains given by the position relation of object bounding boxes.


# Logical Graph Forms

Logical formulas are in the general form when arbitrary, but some can be compiled into other simplified graph forms, with some of them with appealing or simplifying characteristics (satisfiability, determinism, decomposability, polytime satisfiabily, polytime counting...).

#### The Conjunctive Normal Form (CNF)

The CNF is a conjunction (AND) of clauses, where a clause is a disjunction (OR) of literals. A literal is a propositional variable / predicate symbol, possibly preceded by a negation (NOT).

- In the case of the VRD dataset, formula can be directly written in the CNF form. A clause is an imply statement: "Person X wears glasses" implies "glasses are IN Person X" and an imply statement $$ P \rightarrow Q $$ can be written in the form of a disjunction $$ \neg P \lor Q $$.

- In the case of the synthetic dataset, the conversion of arbitrary formulas is performed with `sympy.logic.to_cnf` function from the Sympy package.

The result is saved into [DIMACS format](http://www.satcompetition.org/2009/format-benchmarks2009.html).

#### The decision-Deterministic Decomposable Negation Normal Form (d-DNNF)

The d-DNNF satisfies two properties:

- determinism, requiring operands of an OR to be mutually inconsistent

- decomposability, requiring operands of an AND to be mutually disjoint variables

Conversion from CNF form in DIMACS format to d-DNNF is performed with `c2d_linux` command (in `model.Misc.Formula.dimacs_to_cnf`) of the [C2D compiler from UCLA](http://reasoning.cs.ucla.edu/c2d/).


# Graph embedding

#### Graph Convolution Networks

A stack of multiple Graph Convolution Networks (GCN) is applied to all nodes of a graph of any form. The output of the stack is NxD, where N is the number of graph nodes, and D the dimension of the last layer.

In order to output a global representation of a graph of dimension (D,), a global node is added to the graph, with type global, and linked to all other nodes in the graph: the embedding of this node of dimension D is taken as the **graph embedding**.

The graph definition, input to GCN, is defined by 2 text files:
- a '.var' file, listing all nodes with their ID, features and label
- an '.rel' file, listing all edges between nodes

In the case of CNF, d-DNNF or assignments, the features for a negated literal '-i' is assigned with the negative vector of the feature of the literal `-feature(i)`.

#### Satisfying assignments

Positive assignments (propositions that make the formula True) can also be easily represented by a graph directly in the CNF format:

<img src="{{ site.url }}/img/Assignment.jpg">

corresponding to

$$ \underset{\land}{\text{var} \in \text{Variables}} \( \text{var} \)$$

where each clause is composed of only one variable.

It is then possible to use the **graph embedder** to compute an embedding representation for all assignments and compare them with the formula. During training, embedding of positive assignments are pushed to be closer to embedding of the formula than negative assignments are, thanks to a margin triplet loss based on the euclidian distance:

$$ \| \text{embedding}(\text{formula}) - \text{embedding}(\text{assignment}) \|_2^2 $$

are easier to search from the CNF format with the Solver from the PySat package. Clauses of CNF format are quite simple to express:

- in the VRD dataset, each clause can be expressed in the code with a couple `[-rel_id, pos_id]` where `rel_id` is the ID of a relation `[relation predicate, subject, object]` and `pos_id` is the ID of a spatial property `[position relation, subject, object]`. Variable ID are provided by variable ID manager `pysat.formula.IDPool`, always starting from 1.

- In the case of the synthetic dataset, a `get_clauses()` method parses the CNF clause list to return a list to map index to symbols (`atom_mapping`, for example `['e', 'f']` for a formula with e and f symbols) and a list of clauses in the format of lists as well: [-i] for a Not node, [i] for a symbol, and [(-)i, (-)j, (-)k, ...] where i, j, k are symbol indexes, (-) the Not operator.

5 positive assignments (propositions that make the formula True) are found with the Solver from the PySat package when the formula is of type `pysat.formula.CNF`. 5 negative assignments are easier to find by random tests.

Assignments are considered also as graphs, with one AND node and literals 'i' or '-i'.



Synthetic

VRD RelevantFormulatContainer
clauses and assumptions


??? save assignments, positive, and negatives. Each element a graph



#### Features of the formula graph nodes

In the case of the synthetic dataset, node features come from `model/pygcn/pygcn/features.pk` file containing

- a list `digit_to_sym` to map index to symbol `[None, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']` (None to have all symbol indexes greater than 0)

- a `type_map` dictionary to map type to index: `{'Global': 0, 'Symbol': 1, 'Or': 2, 'And': 3, 'Not': 4}`

- a `features` dictionary containing a feature for each type and symbol `Global, Symbol, Or, And, Not, a, b, c, d, e, f, g, h, i, j, k, l`. The feature is a numpy array of dimension (50,).



In the case of the VRD datasets, for each subject-object pair, a feature vector is created by concatenating

- the image feature: a crop of the union of both bounding boxes is resized to 224x224, normalized, processed by the CV network to produce visual features from before last layer of ResNet 18.

- word embeddings for both subject and object labels. `word_embed.json` contains word embeddings (dimension 300) for all object names from Word2Vec.

- relative position coordinates of subject and object in the crop union of both bounding boxes.

The annotation key is given by

str((subject_category, subject_boundingbox),(object_category, object_boundingbox))

The concepts of subject and object are exchangeable, and a "no-predicate" label is added to all void relations.

The dataset is scanned to search for all position relations possible for each relation `[predicate, subject, object]`, among one of 10 possible exclusive names:
```
'_out', '_in', '_o_left', '_o_right', '_o_up', '_o_down', '_n_left','_n_right', '_n_up', '_n_down'
```
and build the imply clause $$ \text{predicate} \rightarrow \lor \( \lor \text{position} ) $$ for each predicate.

On top of that, for each subject-object pairs, the equivalent reverse position relation clause is added:

```
[['_out', '_in'], ['_o_left', '_o_right'], ['_o_up', '_o_down'],['_n_left', '_n_right'], ['_n_up', '_n_down']]
```

$$ \text{S in O} \rightarrow \text{O out of S}$$

Now that the logic of relations (predicate and relative position) is computed, only clauses that apply to two objects available in image are kept.

??? for CNF conversion : And IDs of relation variables are remapped to a range of values from 1 to N.

For nodes AND, OR, Global, features are reused from synthetic dataset. For other leaf nodes, the features is the average of Glove vectors for all words in the predicate, subjet and object names.

??? different dimensions between 50 and 300
?? problem in averaging

???
```
'_exists'
'_unique'
```

#### Training data

#### Batches of 5 triplets

An object `MyDataset` is implemented for the interface of `torch.utils.data.DataLoader` to deliver batches.

Each batch is composed of 5 items, where each item is a triplet of 3 elements (A, P, N): the anchor (the formula), the positive (or satisfying) assignment, a negative assignment. The three elements are used in the triplet margin loss: the positive assignment has to be closer to the formula than the negative assignment in the embedding space.


### Element loading

Each elements are loaded from file with `load_data` function. It loads:

- node features (each line contains the id, features, label for each node)

??? features
??? labels

- edges, converted into an adjacency matrix A, normalized with

$$ D^{-1} A $$

contrary to paper explanation:

$$ D^{-1/2} A D^{-1/2} $$

The fact A is symetric $$ a_{i,j} = a_{j,i} $$ (undirected graph) does not mean nodes i and j have symetric roles.

The original ID are remapped to 0...N in the order the features are saved.

The return of the function composed of :
- adj: the NxN adjacency matrix in torch sparse float tensor,
- features: a dense float tensor with the features NxD,
- labels: a Nx1 dense long tensor,
- idx_train, idx_val, idx_test: NOT USED (simple range(0,N)),
- and_children, or_children: JSON load from file

??? and_children idx are remapped

Each element is in a graph whose embedding Q of dimension Nx100 (N number of nodes in the graph) is computed with the model, a stack of 4 Graph Convolutions $$A \cdot X \cdot W + B$$ where W are specialized depending on the type of node.

The embedding is trained with triplet margin loss with euclidian distance, plus a regularization loss.

The regularization loss takes the 100-dimension embedding $$ q_i $$ of each children nodes of a AND or OR node and applies:

...

??? difference from paper

A second network, a MLP, is trained to discriminate the embeddings $$ (Q_A, Q_P) $$ and $$ (Q_A, Q_N) $$ with cross entropy loss on top of that.

??? number of node can be different

??? if relation == 100

# Relation prediction

For the VRD experiment, a two-layer MLP is trained to predict the relation. The input of the MLP consists of the image features concatenated with Glove vectors for subject/object labels and relative position coordinates in the image crop. The output is a logit of dimension 71.

First, a Softmax+Crossentropy loss trains the network to predict the relation. Second, thanks to a trained GCN to produce embeddings for the formula and positive assignment to be close, the GCN embedding for the softmax of the logits is constrained to be close to the formula's embedding as well.

Optimizer applies gradients on the MLP model only, contraining only the MLP weights for the ouput to follow the embedder loss.

<img src="{{ site.url }}/img/VRD_clause.jpg">

1 image per batch
batch size of batch == nb relations + some negative subsampling

the feature used for the symbol is an average of prob * feature of the relation name  and embeddings

<img src="{{ site.url }}/img/Assignment_vrd.jpg">
