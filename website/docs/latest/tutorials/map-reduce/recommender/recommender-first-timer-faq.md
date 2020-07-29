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
layout: doc-page
title: (Deprecated)  Recommender First-Timer FAQ

    
---

# Recommender First Timer Dos and Don'ts

Many people with an interest in recommenders arrive at Mahout since they're
building a first recommender system. Some starting questions have been
asked enough times to warrant a FAQ collecting advice and rules-of-thumb to
newcomers.

For the interested, these topics are treated in detail in the book [Mahout in Action](http://manning.com/owen/).

Don't start with a distributed, Hadoop-based recommender; take on that
complexity only if necessary. Start with non-distributed recommenders. It
is simpler, has fewer requirements, and is more flexible. 

As a crude rule of thumb, a system with up to 100M user-item associations
(ratings, preferences) should "fit" onto one modern server machine with 4GB
of heap available and run acceptably as a real-time recommender. The system
is invariably memory-bound since keeping data in memory is essential to
performance.

Beyond this point it gets expensive to deploy a machine with enough RAM,
so, designing for a distributed makes sense when nearing this scale.
However most applications don't "really" have 100M associations to process.
Data can be sampled; noisy and old data can often be aggressively pruned
without significant impact on the result.

The next question is whether or not your system has preference values, or
ratings. Do users and items merely have an association or not, such as the
existence or lack of a click? or is behavior translated into some scalar
value representing the user's degree of preference for the item.

If you have ratings, then a good place to start is a
GenericItemBasedRecommender, plus a PearsonCorrelationSimilarity similarity
metric. If you don't have ratings, then a good place to start is
GenericBooleanPrefItemBasedRecommender and LogLikelihoodSimilarity.

If you want to do content-based item-item similarity, you need to implement
your own ItemSimilarity.

If your data can be simply exported to a CSV file, use FileDataModel and
push new files periodically.
If your data is in a database, use MySQLJDBCDataModel (or its "BooleanPref"
counterpart if appropriate, or its PostgreSQL counterpart, etc.) and put on
top a ReloadFromJDBCDataModel.

This should give a reasonable starter system which responds fast. The
nature of the system is that new data comes in from the file or database
only periodically -- perhaps on the order of minutes. 