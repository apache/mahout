---
layout: default
title: Mahout Benchmarks

    
---

<a name="MahoutBenchmarks-Introduction"></a>
# Introduction

Depending on hardware configuration, exact distribution of ratings over users and items YMMV!

<a name="MahoutBenchmarks-Recommenders"></a>
# Recommenders

<a name="MahoutBenchmarks-ARuleofThumb"></a>
## A Rule of Thumb

100M preferences are about the data set size where non-distributed
recommenders will outgrow a normal-sized machine (32-bit, <= 4GB RAM). Your
mileage will vary significantly with the nature of the data.

<a name="MahoutBenchmarks-Distributedrecommendervs.Wikipedialinks(May272010)"></a>
## Distributed recommender vs. Wikipedia links (May 27 2010)

From the mailing list:

I just finished running a set of recommendations based on the Wikipedia
link graph, for book purposes (yeah, it's unconventional). I ran on my
laptop, but it ought to be crudely representative of how it runs in a real
cluster.

The input is 1058MB as a text file, and contains, 130M article-article
associations, from 5.7M articles to 3.8M distinct articles ("users" and
"items", respectively). I estimate cost based on Amazon's North
American small Linux-based instance pricing of $0.085/hour. I ran on a
dual-core laptop with plenty of RAM, allowing 1GB per worker, so this is
valid.

In this run, I run recommendations for all 5.7M "users". You can certainly
run for any subset of all users of course.

Phase 1 (Item ID to item index mapping)
29 minutes CPU time
$0.05
60MB output

Phase 2 (Create user vectors)
88 minutes CPU time
$0.13
Output: 1159MB

Phase 3 (Count co-occurrence)
77 hours CPU time
$6.54
Output: 23.6GB

Phase 4 (Partial multiply prep)
10.5 hours CPU time
$0.90
Output: 24.6GB

Phase 5 (Aggregate and recommend)
about 600 hours
about $51.00
about 10GB
(I estimated these rather than let it run at home for days!)


Note that phases 1 and 3 may be run less frequently, and need not be run
every time. But the cost is dominated by the last step, which is most of
the work. I've ignored storage costs.

This implies a cost of $0.01 (or about 8 instance-minutes) per 1,000 user
recommendations. That's not bad if, say, you want to update recs for you
site's 100,000 daily active users for a dollar.

There are several levers one could pull internally to sacrifice accuracy
for speed, but it's currently set to pretty normal values. So this is just
one possibility.

Now that's not terrible, but it is about 8x more computing than would be
needed by a non-distributed implementation *if* you could fit the whole
data set into a very large instance's memory, which is still possible at
this scale but needs a pretty big instance. That's a very apples-to-oranges
comparison of course; different algorithms, entirely different
environments. This is about the amount of overhead I'd expect from
distributing -- interesting to note how non-trivial it is.

<a name="MahoutBenchmarks-Non-distributedrecommendervs.KDDCupdataset(March2011)"></a>
## Non-distributed recommender vs. KDD Cup data set (March 2011)

(From the user@mahout.apache.org mailing list)

I've been test-driving a simple application of Mahout recommenders (the
non-distributed kind) on Amazon EC2 on the new Yahoo KDD Cup data set
(kddcup.yahoo.com).

In the spirit of open-source, like I mentioned, I'm committing the extra
code to mahout-examples that can be used to run a Recommender on the input
and output the right format. And, I'd like to publish the rough timings
too. Find all the source in org.apache.mahout.cf.taste.example.kddcup

<a name="MahoutBenchmarks-Track1"></a>
### Track 1

* m2.2xlarge instance, 34.2GB RAM / 4 cores
* Steady state memory consumption: ~19GB
* Computation time: 30 hours (wall clock-time)
* CPU time per user: ~0.43 sec
* Cost on EC2: $34.20 (!)

(Helpful hint on cost I realized after the fact: you can almost surely get
spot instances for cheaper. The maximum price this sort of instance has
gone for as a spot instance is about $0.60/hour, vs "retail price" of
$1.14/hour.)

Resulted in an RMSE of 29.5618 (the rating scale is 0-100), which is only
good enough for 29th place at the moment. Not terrible for "out of the box"
performance -- it's just using an item-based recommender with uncentered
cosine similarity. But not really good in absolute terms. A winning
solution is going to try to factor in time, and apply more sophisticated
techniques. The best RMSE so far is about 23.

<a name="MahoutBenchmarks-Track2"></a>
### Track 2

* c1.xlarge instance: 7GB RAM / 8 cores
* Steady state memory consumption: ~3.8GB
* Computation time: 4.1 hours (wall clock-time)
* CPU time per user: ~1.1 sec
* Cost on EC2: $3.20

For this I bothered to write a simplistic item-item similarity metric to
take into account the additional info that is available: track, artist,
album, genre. The result was comparatively better: 17.92% error rate, good
enough for 4th place at the moment.

Of course, the next task is to put this through the actual distributed
processing -- that's really the appropriate solution.

This shows you can still tackle fairly impressive scale with a
non-distributed solution. These results suggest that the largest instances
available from EC2 would accommodate almost 1 billion ratings in memory.
However at that scale running a user's full recommendations would easily be
measured in seconds, not milliseconds.

<a name="MahoutBenchmarks-Clustering"></a>
# Clustering

See [MAHOUT-588](https://issues.apache.org/jira/browse/MAHOUT-588)


