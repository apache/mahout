---
layout: post
title: Meeting Minutes
date:   2024-11-01 00:00:00 -0800
category: minutes
---
## Weekly community meeting
[Subscribe](mailto:user-subscribe@mahout.apache.org) to the Mahout User list to ask for details on joining.

### Attendees
* Andrew Musselman
* Trevor Grant

### Old Business
* [Splitting Mahout project code into discrete repos (andrew)](https://issues.apache.org/jira/projects/MAHOUT/issues/MAHOUT-2204)
    * mahout-website
    * mahout-classic
    * mahout-samsara
    * mahout-qumat
* [Docker image for web site build (to jowanza)](https://issues.apache.org/jira/projects/MAHOUT/issues/MAHOUT-2165)
* Roadmap
    * Q2
        * Classic in maintenance mode (done)
    * Q3
        * Qumat with hardened (tests, docs, CI/CD) cirq and qiskit backends (in flight)
        * Kernel methods (in flight)
        * Submit public talk about Qumat (done, Fossy Aug 2024)
        * Amazon Braket (done)
    * Q4 and beyond
        * Distributed quantum solvers
* Add GitHub Project
* Add "talks" page to web site (for Andrew)
* PyPi release for QuMat
* Tommy has picked up [kernel methods spike](https://issues.apache.org/jira/browse/MAHOUT-2200) now [in github issue](https://github.com/apache/mahout/issues/456)
    * [457](https://github.com/apache/mahout/issues/457)
    * [458](https://github.com/apache/mahout/issues/458)
    * [Data encoding and kernel notes](https://github.com/apache/mahout/wiki/Data-Encoding-and-Kernel-Notes)
* Reviewing https://github.com/rawkintrevo/mahout/blob/pqc-docs/docs/p_q_c.md
* Reviewing https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.ZZFeatureMap

### New Business
Trevor and Andrew taking these
* [#461 Implement testing workflow](https://github.com/apache/mahout/issues/461)
* [#463 Move PQC docs to this repo](https://github.com/apache/mahout/issues/463)
* [#468 Add a Parameter object](https://github.com/apache/mahout/issues/468)
* [#469 Bind parameter values at execution](https://github.com/apache/mahout/issues/469)
* [#470 Add Qumat to PyPi](https://github.com/apache/mahout/issues/470)

