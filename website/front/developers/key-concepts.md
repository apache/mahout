---
layout: default
title: Key Concepts Overview
theme: 
    name: mahout2
---


Stub:
## Mahout-Samsara Mathematically Expressive Scala DSL

High level over view of how user creates DRMs (which are actually wrappers around underlying bindings data structure)
How Samsara gives R-Like syntax to these DRMs with operations like `drmA.t %*% drmA`.  How the spirit of this is to let 
practitioners quickly develop their own distributed algorithms. 

## Distributed Bindings

Here we'll talk a bit how the user can write distributed bindings for any engine they wish, how they must implement a few
linear algebra operations on the distributed engine in question. 

## Native Solvers

How in JVM based distributed engines, computations happens at JVM on node, native solvers tell application how to dump 
out of JVM and calculate natively, then load back into JVM for shipping. 


## Linear Algebra Algorithms

How algos like dssvd dspca dqr, etc make back bone of algos framework.

## Reccomenders

Mahout's long legacy as leader in Reccomenders in big data, and what is available today.

## Distributed Statistics / Machine Learning Algos a.k.a. pre-canned algos.

How we recognize that not everyone wants to re-invent K-means and linear regression so we are building up a collection of 
common and essoteric algorithms that will come 'pre-canned'

## Map Reduce

How these are legacy but still exist. 

