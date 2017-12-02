---
layout: default
title: How To Update The Website
theme:
    name: retro-mahout
---

# How to update the Mahout Website

Website updates are handled by updating code in the trunk.

You will find markdown pages in `mahout/website`.

Jenkins rebuilds and publishes the website whenever a change is detected in master.

`mahout/website/build_site.sh` contains the script that is used to do this.
