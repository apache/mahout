---
layout: post
title: Meeting Minutes
date:   2023-03-29 00:00:00 -0800
category: minutes
---
## Monthly community call

### Attendees
* Andrew Musselman
* Trevor Grant

### Intros
Round the room quick introductions

### Recap from last month
1. Fixed: [Build failed on AArch64, Fedora 33](https://issues.apache.org/jira/browse/MAHOUT-2139)
1. Fixed: [Fix broken ASF link in Sidebar](https://issues.apache.org/jira/browse/MAHOUT-2161)
1. Fixed: [Edit website](https://issues.apache.org/jira/browse/MAHOUT-2158)
1. New committer Jowanza Joseph (jowanza)
1. New PMC member Shannon Quinn (squinn)

### JIRA focus
1. Assigned: [Web site check](https://issues.apache.org/jira/browse/MAHOUT-2152)
1. Assigned: [Download page improvements](https://issues.apache.org/jira/browse/MAHOUT-2153)
1. [Update NOTICE](https://issues.apache.org/jira/browse/MAHOUT-2154)
1. In progress: [Migrate off Travis](https://issues.apache.org/jira/browse/MAHOUT-2149)
1. Assigned: [Add "how to do your first JIRA" notes](https://issues.apache.org/jira/browse/MAHOUT-2156)
1. Assigned: [Add "first ticket" tag or type in JIRA](https://issues.apache.org/jira/browse/MAHOUT-2157)
1. Assigned: [Add Docker instructions to nav](https://issues.apache.org/jira/browse/MAHOUT-2159)
1. Assigned: [Docker image for web site build](https://issues.apache.org/jira/browse/MAHOUT-2165)


### Release plans
* Apr: 14.2, point release with minor improvements and new build system

### Ideas
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

### Project management
* Inactive PMC

### Summary and next steps
* Continue monthly community calls
* Complete JIRA focus
* Get travis migration done, nightlies running
* Push 14.2 out
* Reevaluate timeline of Ideas in April
