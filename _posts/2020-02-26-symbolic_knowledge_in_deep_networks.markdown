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

Logical formulas are compiled into different graph forms:

- **Conjunctive Normal Form** (CNF) form which is a conjunction (AND) of clauses, where a clause is a disjunction (OR) of literals and a literal is propositional variable / predicate symbol, possibly preceded by a negation (NOT).

In the case of the VRD dataset, a clause is an imply statement: "X wears glasses" implies "glasses are IN X". An imply statement $$ P \rightarrow Q $$ can be written in the form of a disjunction $$ \neg P \lor Q $$. Since the formula are quite simple, in the code, each clause can be expressed as a list `[-rel_id, pos_id]` where `rel_id` is the ID of a relation and `pos_id` is the ID of a spatial property.

In the case of the synthetic dataset, the conversion of arbitrary formulas is performed with `sympy.logic.to_cnf` function from the Sympy package.

The result is saved into [DIMACS format](http://www.satcompetition.org/2009/format-benchmarks2009.html).

- Conversion from CNF form in DIMACS format to decision-Deterministic Decomposable Negation Normal Form (d-DNNF) is performed with `c2d_linux` command (in `model.Misc.Formula.dimacs_to_cnf`) of the [C2D compiler from UCLA](http://reasoning.cs.ucla.edu/c2d/).
