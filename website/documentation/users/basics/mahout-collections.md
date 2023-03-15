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
title: mahout-collections

    
---

# Mahout collections

<a name="mahout-collections-Introduction"></a>
## Introduction

The Mahout Collections library is a set of container classes that address
some limitations of the standard collections in Java. [This presentation](http://domino.research.ibm.com/comm/research_people.nsf/pages/sevitsky.pubs.html/$FILE/oopsla08%20memory-efficient%20java%20slides.pdf)
 describes a number of performance problems with the standard collections. 

Mahout collections addresses two of the more glaring: the lack of support
for primitive types and the lack of open hashing.

<a name="mahout-collections-PrimitiveTypes"></a>
## Primitive Types

The most visible feature of Mahout Collections is the large collection of
primitive type collections. Given Java's asymmetrical support for the
primitive types, the only efficient way to handle them is with many
classes. So, there are ArrayList-like containers for all of the primitive
types, and hash maps for all the useful combinations of primitive type and
object keys and values.

These classes do not, in general, implement interfaces from *java.util*.
Even when the *java.util* interfaces could be type-compatible, they tend
to include requirements that are not consistent with efficient use of
primitive types.

<a name="mahout-collections-OpenAddressing"></a>
# Open Addressing

All of the sets and maps in Mahout Collections are open-addressed hash
tables. Open addressing has a much smaller memory footprint than chaining.
Since the purpose of these collections is to avoid the memory cost of
autoboxing, open addressing is a consistent design choice.

<a name="mahout-collections-Sets"></a>
## Sets

Mahout Collections includes open hash sets. Unlike *java.util*, a set is
not a recycled hash table; the sets are separately implemented and do not
have any additional storage usage for unused keys.

<a name="mahout-collections-CreditwhereCreditisdue"></a>
# Credit where Credit is due

The implementation of Mahout Collections is derived from [Cern Colt](http://acs.lbl.gov/~hoschek/colt/)
.






