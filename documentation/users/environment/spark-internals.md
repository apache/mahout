<!--
 Licensed to the Apache Software Foundation (ASF) under one or more
 contributor license agreements.  See the NOTICE file distributed with
 this work for additional information regarding copyright ownership.
 The ASF licenses this file to You under the Apache License, Version 2.0
 (the "License"); you may not use this file except in compliance with
 the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->
---
layout: default
title: 

   
---

# Introduction

This document provides an overview of how the Mahout Scala DSL (distributed algebraic operators) is implemented over the Spark back end engine. The document is aimed at Mahout developers, to give a high level description of the design. 

## Spark Overview

## Spark Data Model


## Mahout DRM

Mahout DRM, or Distributed Row Matrix, is an abstraction for storing a large matrix of numbers in-memory in a cluster by distributing logical rows among servers. The DSL provides an abstract API on DRMs for backend engines to provide implementations of this API. Examples are Spark and H2O backend engines. Each engine has its own design of mapping the abstract API onto its data model and provide implementations for algebraic operators over that mapping.


## Spark DSL Engine


## Source Layout
