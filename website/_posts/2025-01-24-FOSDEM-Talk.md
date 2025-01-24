---
layout: post
title: Mahout at FOSDEM: "Introducing Qumat"
date:   2025-01-24 00:00:00 -0800
category: news
---
## FOSDEM 2025, ULB Solbosch Campus
## Feb 2nd, Brussels, Belgium
[Introducing Qumat! (An Apache Mahout Joint)](https://fosdem.org/2025/schedule/event/fosdem-2025-5298-introducing-qumat-an-apache-mahout-joint-)

There seem to be as many quantum computing languages as there are quantum computers. IBM’s qiskit attempts to address this by providing interfaces to multiple backends, but do we really want a vendor owning the coding ecosystem? What could go wrong? Lulz. Apache Mahout’s Qumat project allows users to write their circuits once and then run the same code on multiple vendors. In this talk we’ll discuss how Apache Mahout’s Samsara project introduced the idea of avoiding vendor lock-in by allowing machine learning experts to author algorithms in a vendor-neutral language that would then run everywhere. This spirit is living on in the Qumat project which allows user to write quantum circuits and algorithms (in a vendor-neutral Python library), which are then transpiled to run on IBM’s qiskit, Google cirq, Amazon Braket, or to extend the library to run on other hardware.

### Presented by
* Trevor Grant
* Andrew Musselman
