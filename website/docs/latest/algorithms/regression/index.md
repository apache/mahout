---
layout: doc-page

title: Regression Algorithms

    
---

Apache Mahout implements the following regression algorithms "off the shelf".

### Closed Form Solutions

These methods used close form solutions (not stochastic) to solve regression problems

[Ordinary Least Squares](ols.html)

### Autocorrelation Regression

Serial Correlation of the error terms can lead to biased estimates of regression parameters, the following remedial procedures are provided:

[Cochrane Orcutt Procedure](serial-correlation/cochrane-orcutt.html)

[Durbin Watson Test](serial-correlation/dw-test.html)
