---
title: Meeting Minutes
date: 2024-03-07
tags: [minutes]
authors: [mahout-team]
---
## Weekly community meeting

### Attendees
* Andrew Musselman
* Trevor Grant
* Tommy Naugle

### Old Business

### New Business
1. Trevor going off on a spike for kernel methods- what is output?
  * Other gates needed?
  * Other <other> needed?
  * https://chat.openai.com/share/ba29ef75-4158-4e22-be40-78ae14c25f93
1. Coordinate on JIRA
  * Web site cleanup (~200 broken links fixed out of ~220, tommy continuing)
  * Python packaging (jowanza tagged on https://issues.apache.org/jira/browse/MAHOUT-2174) (done)
  * Continued qumat data structure work (tommy in flight, akm to review)
1. Rebuild JIRA - now that we have wiped it clean, on the qumat side anyway, lets start grooming tasks into the appropriate components/releases/etc (todo)
  * Including adding filters to all boards so only those tickets show up (todo)


### Other Business
1. Upgrading Scala and Spark
  * At its core would mean fixing docstrings
  * So it could compile for Java 1.8
  * So we could compile for Spark 2.3/Scala 2.12
  * Or in the mean time use maven flag to ignore docstrings for a local build
1. Ask INFRA to help us make sure PRs are defaulting to main instead of trunk (akm)
1. Make ticket to add notebooks to notebooks directory in source tree (https://issues.apache.org/jira/browse/MAHOUT-2198)
1. JIRA hygiene with Shashanka in monthly Classic meeting (on hold until Shashanka or Eric picks this up, meeting cancelled otherwise)
