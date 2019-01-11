/**
 * <h2>Partial-data mapreduce implementation of Random Decision Forests</h2>
 *
 * <p>The builder splits the data, using a FileInputSplit, among the mappers.
 * Building the forest and estimating the oob error takes two job steps.</p>
 *
 * <p>In the first step, each mapper is responsible for growing a number of trees with its partition's,
 * loading the data instances in its {@code map()} function, then building the trees in the {@code close()} method. It
 * uses the reference implementation's code to build each tree and estimate the oob error.</p>
 *
 * <p>The second step is needed when estimating the oob error. Each mapper loads all the trees that does not
 * belong to its own partition (were not built using the partition's data) and uses them to classify the
 * partition's data instances. The data instances are loaded in the {@code map()} method and the classification
 * is performed in the {@code close()} method.</p>
 */
package org.apache.mahout.classifier.df.mapreduce.partial;
