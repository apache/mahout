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
