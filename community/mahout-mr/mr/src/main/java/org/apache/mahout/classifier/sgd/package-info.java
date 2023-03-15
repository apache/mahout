/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * <p>Implements a variety of on-line logistric regression classifiers using SGD-based algorithms.
 * SGD stands for Stochastic Gradient Descent and refers to a class of learning algorithms
 * that make it relatively easy to build high speed on-line learning algorithms for a variety
 * of problems, notably including supervised learning for classification.</p>
 *
 * <p>The primary class of interest in the this package is
 * {@link org.apache.mahout.classifier.sgd.CrossFoldLearner} which contains a
 * number (typically 5) of sub-learners, each of which is given a different portion of the
 * training data.  Each of these sub-learners can then be evaluated on the data it was not
 * trained on.  This allows fully incremental learning while still getting cross-validated
 * performance estimates.</p>
 *
 * <p>The CrossFoldLearner implements {@link org.apache.mahout.classifier.OnlineLearner}
 * and thus expects to be fed input in the form
 * of a target variable and a feature vector.  The target variable is simply an integer in the
 * half-open interval [0..numFeatures) where numFeatures is defined when the CrossFoldLearner
 * is constructed.  The creation of feature vectors is facilitated by the classes that inherit
 * from {@link org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder}.
 * These classes currently implement a form of feature hashing with
 * multiple probes to limit feature ambiguity.</p>
 */
package org.apache.mahout.classifier.sgd;
