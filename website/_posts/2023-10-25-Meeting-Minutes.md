---
layout: post
title: Meeting Minutes
date:   2023-10-25 00:00:00 -0800
category: minutes
---
## Monthly community meeting

### Attendees
* Andrew Musselman
* Shannon Quinn
* Jowanza Joseph
* Trevor Grant

### JIRA focus
1. TBD: Updating Docker images
1. TBD: Live in-browser code environment
1. TBD: Simple spike for matrix math on the IBM quantum compute platform
1. TBD: Assess all quantum platforms vis-a-vis interop and portability
1. Assigned: [Web site check](https://issues.apache.org/jira/browse/MAHOUT-2152)
1. Assigned: [Download page improvements](https://issues.apache.org/jira/browse/MAHOUT-2153)
1. [Update NOTICE](https://issues.apache.org/jira/browse/MAHOUT-2154)
1. Assigned: [Add "how to do your first JIRA" notes](https://issues.apache.org/jira/browse/MAHOUT-2156)
1. Assigned: [Add "first ticket" tag or type in JIRA](https://issues.apache.org/jira/browse/MAHOUT-2157)
1. Assigned: [Add Docker instructions to nav](https://issues.apache.org/jira/browse/MAHOUT-2159)
1. Assigned: [Docker image for web site build](https://issues.apache.org/jira/browse/MAHOUT-2165)
1. Complete: [Migrate off Travis](https://issues.apache.org/jira/browse/MAHOUT-2149)

### Release plans
* Building web site and library code in GitHub Actions now
* Nightly builds working
* Nov: 14.2, point release with minor improvements and new build system

### Ideas
* Quantum computing (IBM and Honeywell have free-tier APIs) XXXL
* Python bindings for DSL (XXXL)
* Automating or maintaining Docker images (XL, including docs)
* New parsing and processing modules
  1. Proof of concept blockchain data and metadata (XXL)
  1. Proof of concept GPL/BSD/ALv2/MIT classifier (start with small tutorial)
* Live code environment (in-browser) to try DSL (XXL for a real version, start with simple env)
* Language modeling
  1. Guaranteed ALv2-licensed training models
  1. Blessed chat/code recs
* Look at OpenNLP for tokenizing and other processing
* New embedding similarity job

### Summary and next steps
* DONE: Get travis migration done, nightlies running
* TODO in Nov: Push 14.2 out
* Reach out to formerly active project members
* JIRA ranching
