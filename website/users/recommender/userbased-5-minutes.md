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
title: User Based Recommender in 5 Minutes

    
---

# Creating a User-Based Recommender in 5 minutes

##Prerequisites

Create a java project in your favorite IDE and make sure mahout is on the classpath. The easiest way to accomplish this is by importing it via maven as described on the [Quickstart](/users/basics/quickstart.html) page.


## Dataset

Mahout's recommenders expect interactions between users and items as input. The easiest way to supply such data to Mahout is in the form of a textfile, where every line has the format *userID,itemID,value*. Here *userID* and *itemID* refer to a particular user and a particular item, and *value* denotes the strength of the interaction (e.g. the rating given to a movie).

In this example, we'll use some made up data for simplicity. Create a file called "dataset.csv" and copy the following example interactions into the file. 

<pre>
1,10,1.0
1,11,2.0
1,12,5.0
1,13,5.0
1,14,5.0
1,15,4.0
1,16,5.0
1,17,1.0
1,18,5.0
2,10,1.0
2,11,2.0
2,15,5.0
2,16,4.5
2,17,1.0
2,18,5.0
3,11,2.5
3,12,4.5
3,13,4.0
3,14,3.0
3,15,3.5
3,16,4.5
3,17,4.0
3,18,5.0
4,10,5.0
4,11,5.0
4,12,5.0
4,13,0.0
4,14,2.0
4,15,3.0
4,16,1.0
4,17,4.0
4,18,1.0
</pre>

## Creating a user-based recommender

Create a class called *SampleRecommender* with a main method.

The first thing we have to do is load the data from the file. Mahout's recommenders use an interface called *DataModel* to handle interaction data. You can load our made up interactions like this:

<pre>
DataModel model = new FileDataModel(new File("/path/to/dataset.csv"));
</pre>

In this example, we want to create a user-based recommender. The idea behind this approach is that when we want to compute recommendations for a particular users, we look for other users with a similar taste and pick the recommendations from their items. For finding similar users, we have to compare their interactions. There are several methods for doing this. One popular method is to compute the [correlation coefficient](https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient) between their interactions. In Mahout, you use this method as follows:

<pre>
UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
</pre>

The next thing we have to do is to define which similar users we want to leverage for the recommender. For the sake of simplicity, we'll use all that have a similarity greater than *0.1*. This is implemented via a *ThresholdUserNeighborhood*:

<pre>UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, model);</pre>

Now we have all the pieces to create our recommender:

<pre>
UserBasedRecommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);
</pre>
        
We can easily ask the recommender for recommendations now. If we wanted to get three items recommended for the user with *userID* 2, we would do it like this:
	

<pre>
List<RecommendedItem> recommendations = recommender.recommend(2, 3);
for (RecommendedItem recommendation : recommendations) {
  System.out.println(recommendation);
}
</pre>


Congratulations, you have built your first recommender!


## Evaluation

You might ask yourself, how to make sure that your recommender returns good results. Unfortunately, the only way to be really sure about the quality is by doing an A/B test with real users in a live system.

We can however try to get a feel of the quality, by statistical offline evaluation. Just keep in mind that this does not replace a test with real users!

One way to check whether the recommender returns good results is by doing a **hold-out** test. We partition our dataset into two sets: a trainingset consisting of 90% of the data and a testset consisting of 10%. Then we train our recommender using the training set and look how well it predicts the unknown interactions in the testset.

To test our recommender, we create a class called *EvaluateRecommender* with a main method and add an inner class called *MyRecommenderBuilder* that implements the *RecommenderBuilder* interface. We implement the *buildRecommender* method and make it setup our user-based recommender:

<pre>
UserSimilarity similarity = new PearsonCorrelationSimilarity(dataModel);
UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, dataModel);
return new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
</pre>

Now we have to create the code for the test. We'll check how much the recommender misses the real interaction strength on average. We employ an *AverageAbsoluteDifferenceRecommenderEvaluator* for this. The following code shows how to put the pieces together and run a hold-out test: 

<pre>
DataModel model = new FileDataModel(new File("/path/to/dataset.csv"));
RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
RecommenderBuilder builder = new MyRecommenderBuilder();
double result = evaluator.evaluate(builder, null, model, 0.9, 1.0);
System.out.println(result);
</pre>

Note: if you run this test multiple times, you will get different results, because the splitting into trainingset and testset is done randomly. 











