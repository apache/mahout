<!--
 Licensed to the Apache Software Foundation (ASF) under one or more
 contributor license agreements.  See the NOTICE file distributed with
 this work for additional information regarding copyright ownership.
 The ASF licenses this file to You under the Apache License, Version 2.0
 (the "License"); you may not use this file except in compliance with
 the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->
---
layout: default
title: Perceptron and Winnow

    
---

# Introduction to ALS Recommendations with Hadoop

##Overview

Mahout’s ALS recommender is a matrix factorization algorithm that uses Alternating Least Squares with Weighted-Lamda-Regularization (ALS-WR). It factors the user to item matrix *A* into the user-to-feature matrix *U* and the item-to-feature matrix *M*: It runs the ALS algorithm in a parallel fashion. The algorithm details can be referred to in the following papers: 

* [Large-scale Parallel Collaborative Filtering for
the Netflix Prize](http://www.hpl.hp.com/personal/Robert_Schreiber/papers/2008%20AAIM%20Netflix/netflix_aaim08%28submitted%29.pdf)
* [Collaborative Filtering for Implicit Feedback Datasets](http://research.yahoo.com/pub/2433) 

This recommendation algorithm can be used in eCommerce platform to recommend products to customers. Unlike the user or item based recommenders that computes the similarity of users or items to make recommendations, the ALS algorithm uncovers the latent factors that explain the observed user to item ratings and tries to find optimal factor weights to minimize the least squares between predicted and actual ratings.

Mahout's ALS recommendation algorithm takes as input user preferences by item and generates an output of recommending items for a user. The input customer preference could either be explicit user ratings or implicit feedback such as user's click on a web page.

One of the strengths of the ALS based recommender, compared to the user or item based recommender, is its ability to handle large sparse data sets and its better prediction performance. It could also gives an intuitive rationale of the factors that influence recommendations.

##Implementation
At present Mahout has a map-reduce implementation of ALS, which is composed of 2 jobs: a parallel matrix factorization job and a recommendation job.
The matrix factorization job computes the user-to-feature matrix and item-to-feature matrix given the user to item ratings. Its input includes: 
<pre>
    --input: directory containing files of explicit user to item rating or implicit feedback;
    --output: output path of the user-feature matrix and feature-item matrix;
    --lambda: regularization parameter to avoid overfitting;
    --alpha: confidence parameter only used on implicit feedback
    --implicitFeedback: boolean flag to indicate whether the input dataset contains implicit feedback;
    --numFeatures: dimensions of feature space;
    --numThreadsPerSolver: number of threads per solver mapper for concurrent execution;
    --numIterations: number of iterations
    --usesLongIDs: boolean flag to indicate whether the input contains long IDs that need to be translated
</pre>
and it outputs the matrices in sequence file format. 

The recommendation job uses the user feature matrix and item feature matrix calculated from the factorization job to compute the top-N recommendations per user. Its input includes:
<pre>
    --input: directory containing files of user ids;
    --output: output path of the recommended items for each input user id;
    --userFeatures: path to the user feature matrix;
    --itemFeatures: path to the item feature matrix;
    --numRecommendations: maximum number of recommendations per user, default is 10;
    --maxRating: maximum rating available;
    --numThreads: number of threads per mapper;
    --usesLongIDs: boolean flag to indicate whether the input contains long IDs that need to be translated;
    --userIDIndex: index for user long IDs (necessary if usesLongIDs is true);
    --itemIDIndex: index for item long IDs (necessary if usesLongIDs is true) 
</pre>
and it outputs a list of recommended item ids for each user. The predicted rating between user and item is a dot product of the user's feature vector and the item's feature vector.  

##Example

Let’s look at a simple example of how we could use Mahout’s ALS recommender to recommend items for users. First, you’ll need to get Mahout up and running, the instructions for which can be found [here](https://mahout.apache.org/users/basics/quickstart.html). After you've ensured Mahout is properly installed, we’re ready to run the example.

**Step 1: Prepare test data**

Similar to Mahout's item based recommender, the ALS recommender relies on the user to item preference data: *userID*, *itemID* and *preference*. The preference could be explicit numeric rating or counts of actions such as a click (implicit feedback). The test data file is organized as each line is a tab-delimited string, the 1st field is user id, which must be numeric, the 2nd field is item id, which must be numeric and the 3rd field is preference, which should also be a number.

**Note:** You must create IDs that are ordinal positive integers for all user and item IDs. Often this will require you to keep a dictionary
to map into and out of Mahout IDs. For instance if the first user has ID "xyz" in your application, this would get an Mahout ID of the integer 1 and so on. The same
for item IDs. Then after recommendations are calculated you will have to translate the Mahout user and item IDs back into your application IDs.

To quickly start, you could specify a text file like following as the input:
<pre>
1	100	1
1	200	5
1	400	1
2	200	2
2	300	1
</pre>

**Step 2: Determine parameters**

In addition, users need to determine dimension of feature space, the number of iterations to run the alternating least square algorithm, Using 10 features and 15 iterations is a reasonable default to try first. Optionally a confidence parameter can be set if the input preference is implicit user feedback.  

**Step 3: Run ALS**

Assuming your *JAVA_HOME* is appropriately set and Mahout was installed properly we’re ready to configure our syntax. Enter the following command:

    $ mahout parallelALS --input $als_input --output $als_output --lambda 0.1 --implicitFeedback true --alpha 0.8 --numFeatures 2 --numIterations 5  --numThreadsPerSolver 1 --tempDir tmp 

Running the command will execute a series of jobs the final product of which will be an output file deposited to the output directory specified in the command syntax. The output directory contains 3 sub-directories: *M* stores the item to feature matrix, *U* stores the user to feature matrix and userRatings stores the user's ratings on the items. The *tempDir* parameter specifies the directory to store the intermediate output of the job, such as the matrix output in each iteration and each item's average rating. Using the *tempDir* will help on debugging.

**Step 4: Make Recommendations**

Based on the output feature matrices from step 3, we could make recommendations for users. Enter the following command:

     $ mahout recommendfactorized --input $als_recommender_input --userFeatures $als_output/U/ --itemFeatures $als_output/M/ --numRecommendations 1 --output recommendations --maxRating 1

The input user file is a sequence file, the sequence record key is user id and value is the user's rated item ids which will be removed from recommendation. The output file generated in our simple example will be a text file giving the recommended item ids for each user. 
Remember to translate the Mahout ids back into your application specific ids. 

There exist a variety of parameters for Mahout’s ALS recommender to accommodate custom business requirements; exploring and testing various configurations to suit your needs will doubtless lead to additional questions. Feel free to ask such questions on the [mailing list](https://mahout.apache.org/general/mailing-lists,-irc-and-archives.html).

