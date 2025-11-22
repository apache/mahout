---
title: "Large-Scale Distributed Linear Algebra and Optimization Using Apache Spark"
authors: ["A. F. Davidson", "J. Rosen", "M. Zaharia"]
year: 2016
link: "https://arxiv.org/abs/1605.08325"
---

### Summary
The paper explores distributed linear algebra primitives built on Apache Spark, focusing on block matrices, optimized shuffles, and distributed optimization routines. It benchmarks matrix multiplication, SVD, and iterative solvers at cluster scale. These methods form the foundation for scalable ML pipelines and align closely with Mahout’s goal of providing backend-agnostic distributed linear algebra via Samsara and Spark bindings.