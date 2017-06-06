---
layout: default
title: (Deprecated)  Testing
theme:
    name: retro-mahout
---
<a name="Testing-Intro"></a>
# Intro

As Mahout matures, solid testing procedures are needed.  This page and its
children capture test plans along with ideas for improving our testing.

<a name="Testing-TestPlans"></a>
# Test Plans

* [0.6](0.6.html)
 - Test Plans for the 0.6 release
There are no special plans except for unit tests, and user testing of the
Hadoop jobs.

<a name="Testing-TestIdeas"></a>
# Test Ideas

<a name="Testing-Regressions/Benchmarks/Integrations"></a>
## Regressions/Benchmarks/Integrations
* Algorithmic quality and speed are not tested, except in a few instances.
Such tests often require much longer run times (minutes to hours), a
running Hadoop cluster, and downloads of large datasets (in the megabytes). 
* Standardized speed tests are difficult on different hardware. 
* Unit tests of external integrations require access to externals: HDFS,
S3, JDBC, Cassandra, etc. 

Apache Jenkins is not able to support these environments. Commercial
donations would help. 

<a name="Testing-UnitTests"></a>
## Unit Tests
Mahout's current tests are almost entirely unit tests. Algorithm tests
generally supply a few numbers to code paths and verify that expected
numbers come out. 'mvn test' runs these tests. There is "positive" coverage
of a great many utilities and algorithms. A much smaller percent include
"negative" coverage (bogus setups, inputs, combinations).

<a name="Testing-Other"></a>
## Other

