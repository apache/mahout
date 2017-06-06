---
layout: default
title: 
theme:
   name: retro-mahout
---

# Introduction

This document provides an overview of how the Mahout Scala DSL (distributed algebraic operators) is implemented over the Spark back end engine. The document is aimed at Mahout developers, to give a high level description of the design. 

## Spark Overview

## Spark Data Model


## Mahout DRM

Mahout DRM, or Distributed Row Matrix, is an abstraction for storing a large matrix of numbers in-memory in a cluster by distributing logical rows among servers. The DSL provides an abstract API on DRMs for backend engines to provide implementations of this API. Examples are Spark and H2O backend engines. Each engine has its own design of mapping the abstract API onto its data model and provide implementations for algebraic operators over that mapping.


## Spark DSL Engine


## Source Layout
