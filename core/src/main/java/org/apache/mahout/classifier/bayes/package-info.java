/**
 * <h2>Introduction</h2>
 *
 * <p>This package provides an implementation of a MapReduce-enabled Naïve Bayes classifier. It
 * is a very simple classifier that counts the occurrences of words in association with a label which
 * can then be used to determine the likelihood that a new document, and its words, should be assigned a particular
 * label.
 * </p>
 *
 * <h2>Implementation</h2>
 *
 * <p>The implementation is divided up into three parts:</p>
 *
 * <ol>
 *   <li>The Trainer -- responsible for doing the counting of the words and the labels</li>
 *   <li>The Model -- responsible for holding the training data in a useful way</li>
 *   <li>The Classifier -- responsible for using the trainers output to determine the category of previously unseen
 *     documents</li>
 * </ol>
 *
 * <h3>The Trainer</h3>

 * <p>The trainer is manifested in several classes:</p>
 *
 * <ol>
 *   <li>{@link org.apache.mahout.classifier.bayes.mapreduce.bayes.BayesDriver} -- Creates the Hadoop Naive Bayes job and outputs
 *     the model. This Driver encapsulates a lot of intermediate Map-Reduce Classes</li>
 *   <li>{@link org.apache.mahout.classifier.bayes.mapreduce.common.BayesFeatureDriver}</li>
 *   <li>{@link org.apache.mahout.classifier.bayes.mapreduce.common.BayesTfIdfDriver}</li>
 *   <li>{@link org.apache.mahout.classifier.bayes.mapreduce.common.BayesWeightSummerDriver}</li>
 *   <li>{@link org.apache.mahout.classifier.bayes.mapreduce.bayes.BayesThetaNormalizerDriver}</li>
 * </ol>
 *
 * <p>The trainer assumes that the input files are in the {@link org.apache.hadoop.mapred.KeyValueTextInputFormat},
 * i.e. the first token of the line is the label and separated from the remaining tokens on the line by a
 * tab-delimiter. The remaining tokens are the unique features (words). Thus, input documents might look like:</p>
 *
 * <pre>
 * hockey puck stick goalie forward defenseman referee ice checking slapshot helmet
 * football field football pigskin referee helmet turf tackle
 * </pre>
 *
 * <p>where hockey and football are the labels and the remaining words are the features associated with those
 * particular labels.</p>
 *
 * <p>The output from the trainer is a {@link org.apache.hadoop.io.SequenceFile}.</p>
 */
package org.apache.mahout.classifier.bayes;