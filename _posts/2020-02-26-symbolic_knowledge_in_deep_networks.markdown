---
layout: post
title:  "Symbolic Knowledge in deep learning"
date:   2020-02-26 05:00:00
categories: deep learning
---

In this post, I'll give some explanations about [Embedding Symbolic Knowledge into Deep Networks](https://arxiv.org/abs/1909.01161) paper with [code](https://github.com/ZiweiXU/LENSR).

In particular:

1/ what are the features X of leaf nodes in the symbolic knowledge embedder ?

2/ How is computed the embedding of the formula ?

3/ How do we link formula satisfaction and model predictions during training ?

The idea of the paper is to use Graph Convolutional Networks (GCN) to embed logical formulas. There exists multiple languages into which a formula can be compiled, with some appealing or simplifying characteristics (satisfiability, determinism, decomposability, polytime satisfiabily, polytime counting...).

# Forms

Logical formulas are compiled into different graph forms:

#### Conjunctive Normal Form (CNF)

The CNF is a conjunction (AND) of clauses, where a clause is a disjunction (OR) of literals. A literal is a propositional variable / predicate symbol, possibly preceded by a negation (NOT).

- In the case of the VRD dataset, a clause is an imply statement: "Person X wears glasses" implies "glasses are IN Person X". An imply statement $$ P \rightarrow Q $$ can be written in the form of a disjunction $$ \neg P \lor Q $$. Since the clauses are quite simple, each clause can be expressed in the code with a couple `[-rel_id, pos_id]` where `rel_id` is the ID of a relation and `pos_id` is the ID of a spatial property.

??? id of a relation

- In the case of the synthetic dataset, the conversion of arbitrary formulas is performed with `sympy.logic.to_cnf` function from the Sympy package.

The result is saved into [DIMACS format](http://www.satcompetition.org/2009/format-benchmarks2009.html).

#### decision-Deterministic Decomposable Negation Normal Form (d-DNNF)

The d-DNNF satisfies two properties:

- determinism, requiring operands of an OR to be mutually inconsistent

- decomposability, requiring operands of an AND to be mutually disjoint variables

Conversion from CNF form in DIMACS format to d-DNNF is performed with `c2d_linux` command (in `model.Misc.Formula.dimacs_to_cnf`) of the [C2D compiler from UCLA](http://reasoning.cs.ucla.edu/c2d/).


# Assignments

5 positive assignments (propositions that make the formula True) are found with the Solver from the PySat package when the formula is of type `pysat.formula.CNF`. 5 negative assignments are easier to find by random tests.

Synthetic get_clauses

VRD RelevantFormulatContainer
clauses and assumptions


??? save assignments, positive, and negatives. Each element a graph


# Graph data format

An object `MyDataset` is implemented for the interface of `torch.utils.data.DataLoader` to deliver batches.

Each batch is composed of 5 items, where each item is a triplet of 3 elements (A, P, N): the anchor (the formula), the positive (or satisfying) assignment, a negative assignment. The three elements are used in the triplet margin loss: the positive assignment has to be closer to the formula than the negative assignment in the embedding space.

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

Each element is in a graph whose embedding Q of dimension Nx100 (N number of nodes in the graph) is computed with the model, a stack of 4 Graph Convolutions $$A \dot X \dot W + B$$ where W are specialized depending on the type of node.

The embedding is trained with triplet margin loss with euclidian distance, plus a regularization loss.

The regularization loss takes the 100-dimension embedding $$ q_i $$ of each children nodes of a AND or OR node and applies:

...

??? difference from paper

A second network, a MLP, is trained to discriminate the embeddings $$ (Q_A, Q_P) $$ and $$ (Q_A, Q_N) $$ with cross entropy loss on top of that.

??? number of node can be different
