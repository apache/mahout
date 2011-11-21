This directory contains helpful shell scripts for working with some of Mahout's examples.  

Here's a description of what each does:

asf-email-examples.sh -- Recommend, classify and cluster the ASF Email Public Dataset, as hosted on Amazon (http://aws.amazon.com/datasets/7791434387204566).  Requires download.  Demonstrates a number of Mahout algorithms.
classify-20newsgroups.sh -- Run SGD and Bayes classifiers over the classic 20 News Groups.  Downloads the data set automatically.
cluster-reuters.sh -- Cluster the Reuters data set using a variety of algorithms.  Downloads the data set automatically.
cluster-syntheticcontrol.sh -- Cluster the Synthetic Control data set.  Downloads the data set automatically.
factorize-movielens-1m.sh -- Run the Alternating Least Squares Recommender on the Grouplens data set (size 1M).
factorize-netflix.sh -- Run the ALS Recommender on the Netflix data set


If you are looking for the build-* scripts (build-asf-email.sh, build-reuters.sh), they have been renamed to better signify what they do.  See https://issues.apache.org/jira/browse/MAHOUT-868 for more information.  These have been renamed to:

build-asf-email.sh -> asf-email-examples.sh
build-cluster-syntheticcontrol.sh -> cluster-syntheticcontrol.sh
build-reuters.sh -> cluster-reuters.sh
