---
layout: default
title:
redirect_from:
  - /documentation/users/classification/bankmarketing-example
  - /documentation/users/classification/bankmarketing-example.html
---

#Bank Marketing Example

### Introduction

This page describes how to run Mahout's SGD classifier on the [UCI Bank Marketing dataset](http://mlr.cs.umass.edu/ml/datasets/Bank+Marketing).
The goal is to predict if the client will subscribe a term deposit offered via a phone call. The features in the dataset consist
of information such as age, job, marital status as well as information about the last contacts from the bank.

### Code & Data

The bank marketing example code lives under

*mahout-examples/src/main/java/org.apache.mahout.classifier.sgd.bankmarketing*

The data can be found at

*mahout-examples/src/main/resources/bank-full.csv*

### Code details

This example consists of 3 classes:

  - BankMarketingClassificationMain
  - TelephoneCall
  - TelephoneCallParser

When you run the main method of BankMarketingClassificationMain it parses the dataset using the TelephoneCallParser and trains
a logistic regression model with 20 runs and 20 passes. The TelephoneCallParser uses Mahout's feature vector encoder
to encode the features in the dataset into a vector. Afterwards the model is tested and the learning rate and AUC is printed accuracy is printed to standard output.
