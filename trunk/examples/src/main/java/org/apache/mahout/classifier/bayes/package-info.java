/**
 * <p>The Bayes example package provides some helper classes for training the Naive Bayes classifier
 * on the Twenty Newsgroups data. See {@link org.apache.mahout.classifier.bayes.PrepareTwentyNewsgroups}
 * for details on running the trainer and
 * formatting the Twenty Newsgroups data properly for the training.</p>
 *
 * <p>The easiest way to prepare the data is to use the ant task in core/build.xml:</p>
 *
 * <p>{@code ant extract-20news-18828}</p>
 *
 * <p>This runs the arg line:</p>
 *
 * <p>{@code -p $\{working.dir\}/20news-18828/ -o $\{working.dir\}/20news-18828-collapse -a $\{analyzer\} -c UTF-8}</p>
 *
 * <p>To Run the Wikipedia examples (assumes you've built the Mahout Job jar):</p>
 *
 * <ol>
 *  <li>Download the Wikipedia Dataset. Use the Ant target: {@code ant enwiki-files}</li>
 *  <li>Chunk the data using the WikipediaXmlSplitter (from the Hadoop home):
 *   {@code bin/hadoop jar $MAHOUT_HOME/target/mahout-examples-0.x
 *   org.apache.mahout.classifier.bayes.WikipediaXmlSplitter
 *   -d $MAHOUT_HOME/examples/temp/enwiki-latest-pages-articles.xml
 *   -o $MAHOUT_HOME/examples/work/wikipedia/chunks/ -c 64}</li>
 * </ol>
 */
package org.apache.mahout.classifier.bayes;
