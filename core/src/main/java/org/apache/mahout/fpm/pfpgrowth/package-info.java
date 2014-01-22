/**
 * <h2>MapReduce (parallel) implementation of FP Growth Algorithm for frequent Itemset Mining</h2>
 *
 * <p>We have a Top K Parallel FPGrowth Implementation. What it means is that given a huge transaction list,
 * we find all unique features(field values) and eliminates those features whose frequency in the whole dataset
 * is less that {@code minSupport}. Using these remaining features N, we find the top K closed patterns for
 * each of them, generating NK patterns. FPGrowth Algorithm is a generic implementation, we can use any object
 * type to denote a feature. Current implementation requires you to use a String as the object type. You may
 * implement a version for any object by creating {@link java.util.Iterator}s, Convertors
 * and TopKPatternWritable for that particular object. For more information please refer the package
 * {@code org.apache.mahout.fpm.pfpgrowth.convertors.string}.</p>
 *
 * {@code
 * FPGrowth<String> fp = new FPGrowth<String>();
 * Set<String> features = new HashSet<String>();
 * fp.generateTopKStringFrequentPatterns(
 *   new StringRecordIterator(
 *     new FileLineIterable(new File(input), encoding, false), pattern),
 *     fp.generateFList(
 *       new StringRecordIterator(new FileLineIterable(new File(input), encoding, false), pattern), minSupport),
 *     minSupport,
 *     maxHeapSize,
 *     features,
 *     new StringOutputConvertor(new SequenceFileOutputCollector<Text,TopKStringPatterns>(writer)));}
 *
 * <ul>
 * <li>The first argument is the iterator of transaction in this case its {@code Iterator<List<String>>}</li>
 * <li>The second argument is the output of generateFList function, which returns the frequent items and
 *  their frequencies from the given database transaction iterator</li>
 * <li>The third argument is the minimum Support of the pattern to be generated</li>
 * <li>The fourth argument is the maximum number of patterns to be mined for each feature</li>
 * <li>The fifth argument is the set of features for which the frequent patterns has to be mined</li>
 * <li>The last argument is an output collector which takes [key, value] of Feature and TopK Patterns of the format
 *  {@code [String, List<Pair<List<String>,Long>>]} and writes them to the appropriate writer class
 *  which takes care of storing the object, in this case in a
 *  {@link org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat}</li>
 * </ul>
 *
 * <p>The command line launcher for string transaction data {@code org.apache.mahout.fpm.pfpgrowth.FPGrowthJob}
 * has other features including specifying the regex pattern for spitting a string line of a transaction into
 * the constituent features.</p>
 *
 * <p>The {@code numGroups} parameter in FPGrowthJob specifies the number of groups into which transactions
 * have to be decomposed. The {@code numTreeCacheEntries} parameter specifies the number of generated
 * conditional FP-Trees to be kept in memory so as not to regenerate them. Increasing this number
 * increases the memory consumption but might improve speed until a certain point. This depends entirely on
 * the dataset in question. A value of 5-10 is recommended for mining up to top 100 patterns for each feature.</p>
 */
package org.apache.mahout.fpm.pfpgrowth;
