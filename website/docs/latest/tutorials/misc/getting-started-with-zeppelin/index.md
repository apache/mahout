---
layout: doc-page
title: Getting Started with Mahout in Apache Zeppelin
  
---

Mahout has lots of pieces, and can be overwhelming to get started. We've tried to make it easier for users by providing
a precompiled Docker container which runs Apache Zeppelin (a popular notebooking tool) with an Apache Spark 
interpreter that is preconfigured for Mahout and has all the required libraries loaded. We hope this will help you get 
"up and running with Mahout" as quickly as possible. (If you are interested in adding Apache Mahout to an existing Zeppelin
Instance, see [this tutorial](http://mahout.apache.org/docs/latest/tutorials/misc/mahout-in-zeppelin/)).

## Running the Container

Running Zeppelin with a preconfigured Mahout+Spark interpreter is a three-step process:

#### 1. Pull the image

In a shell (on a machine with Docker installed), type:
```bash
docker pull apache/mahout-zeppelin:14.1
```

#### 2. Run the container

Next type in the shell:
```bash
docker run -p 8080:8080 --rm --name whatever apache/mahout-zeppelin:14.1
```

#### 3. Surf to your running Zeppelin instance

Open a browser and enter [http://localhost:8080](http://localhost:8080)

## What next?!

How easy was that?! (If you have troubles, make sure to sign up to user@mahout.apache.org, we'll be happy to help you).

Next, check out the notebooks- one gives a nice primer to Apache Mahout already loaded. Enjoy!

