This directory contains helpful shell scripts for working with some of Mahout's examples.  

To set a non-default temporary work directory: `export MAHOUT_WORK_DIR=/path/in/hdfs/to/temp/dir`
  Note that this requires the same path to be writable both on the local file system as well as on HDFS.

Here's a description of what each does:

classify-20newsgroups.sh -- Run SGD and Bayes classifiers over the classic 20 News Groups.  Downloads the data set automatically.
cluster-reuters.sh -- Cluster the Reuters data set using a variety of algorithms.  Downloads the data set automatically.
cluster-syntheticcontrol.sh -- Cluster the Synthetic Control data set.  Downloads the data set automatically.
factorize-movielens-1m.sh -- Run the Alternating Least Squares Recommender on the Grouplens data set (size 1M).
factorize-netflix.sh -- (Deprecated due to lack of availability of the data set) Run the ALS Recommender on the Netflix data set.
spark-document-classifier.mscala -- A mahout-shell script which trains and tests a Naive Bayes model on the Wikipedia XML dump and defines simple methods to classify new text.
