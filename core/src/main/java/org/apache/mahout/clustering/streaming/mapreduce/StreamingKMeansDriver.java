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

package org.apache.mahout.clustering.streaming.mapreduce;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.neighborhood.BruteSearch;
import org.apache.mahout.math.neighborhood.ProjectionSearch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Classifies the vectors into different clusters found by the clustering
 * algorithm.
 */
public final class StreamingKMeansDriver extends AbstractJob {
  /**
   * Streaming KMeans options
   */
  /**
   * The number of cluster that Mappers will use should be \(O(k log n)\) where k is the number of clusters
   * to get at the end and n is the number of points to cluster. This doesn't need to be exact.
   * It will be adjusted at runtime.
   */
  public static final String ESTIMATED_NUM_MAP_CLUSTERS = "estimatedNumMapClusters";
  /**
   * The initial estimated distance cutoff between two points for forming new clusters.
   * @see org.apache.mahout.clustering.streaming.cluster.StreamingKMeans
   * Defaults to 10e-6.
   */
  public static final String ESTIMATED_DISTANCE_CUTOFF = "estimatedDistanceCutoff";

  /**
   * Ball KMeans options
   */
  /**
   * After mapping finishes, we get an intermediate set of vectors that represent approximate
   * clusterings of the data from each Mapper. These can be clustered by the Reducer using
   * BallKMeans in memory. This variable is the maximum number of iterations in the final
   * BallKMeans algorithm.
   * Defaults to 10.
   */
  public static final String MAX_NUM_ITERATIONS = "maxNumIterations";
  /**
   * The "ball" aspect of ball k-means means that only the closest points to the centroid will actually be used
   * for updating. The fraction of the points to be used is those points whose distance to the center is within
   * trimFraction * distance to the closest other center.
   * Defaults to 0.9.
   */
  public static final String TRIM_FRACTION = "trimFraction";
  /**
   * Whether to use k-means++ initialization or random initialization of the seed centroids.
   * Essentially, k-means++ provides better clusters, but takes longer, whereas random initialization takes less
   * time, but produces worse clusters, and tends to fail more often and needs multiple runs to compare to
   * k-means++. If set, uses randomInit.
   * @see org.apache.mahout.clustering.streaming.cluster.BallKMeans
   */
  public static final String RANDOM_INIT = "randomInit";
  /**
   * Whether to correct the weights of the centroids after the clustering is done. The weights end up being wrong
   * because of the trimFraction and possible train/test splits. In some cases, especially in a pipeline, having
   * an accurate count of the weights is useful. If set, ignores the final weights.
   */
  public static final String IGNORE_WEIGHTS = "ignoreWeights";
  /**
   * The percentage of points that go into the "test" set when evaluating BallKMeans runs in the reducer.
   */
  public static final String TEST_PROBABILITY = "testProbability";
  /**
   * The percentage of points that go into the "training" set when evaluating BallKMeans runs in the reducer.
   */
  public static final String NUM_BALLKMEANS_RUNS = "numBallKMeansRuns";

  /**
   Searcher options
   */
  /**
   * The Searcher class when performing nearest neighbor search in StreamingKMeans.
   * Defaults to ProjectionSearch.
   */
  public static final String SEARCHER_CLASS_OPTION = "searcherClass";
  /**
   * The number of projections to use when using a projection searcher like ProjectionSearch or
   * FastProjectionSearch. Projection searches work by projection the all the vectors on to a set of
   * basis vectors and searching for the projected query in that totally ordered set. This
   * however can produce false positives (vectors that are closer when projected than they would
   * actually be.
   * So, there must be more than one projection vectors in the basis. This variable is the number
   * of vectors in a basis.
   * Defaults to 3
   */
  public static final String NUM_PROJECTIONS_OPTION = "numProjections";
  /**
   * When using approximate searches (anything that's not BruteSearch),
   * more than just the seemingly closest element must be considered. This variable has different
   * meanings depending on the actual Searcher class used but is a measure of how many candidates
   * will be considered.
   * See the ProjectionSearch, FastProjectionSearch, LocalitySensitiveHashSearch classes for more
   * details.
   * Defaults to 2.
   */
  public static final String SEARCH_SIZE_OPTION = "searchSize";

  /**
   * Whether to run another pass of StreamingKMeans on the reducer's points before BallKMeans. On some data sets
   * with a large number of mappers, the intermediate number of clusters passed to the reducer is too large to
   * fit into memory directly, hence the option to collapse the clusters further with StreamingKMeans.
   */
  public static final String REDUCE_STREAMING_KMEANS = "reduceStreamingKMeans";

  private static final Logger log = LoggerFactory.getLogger(StreamingKMeansDriver.class);

  public static final float INVALID_DISTANCE_CUTOFF = -1;

  @Override
  public int run(String[] args) throws Exception {
    // Standard options for any Mahout job.
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.overwriteOption().create());

    // The number of clusters to create for the data.
    addOption(DefaultOptionCreator.numClustersOption().withDescription(
        "The k in k-Means. Approximately this many clusters will be generated.").create());

    // StreamingKMeans (mapper) options
    // There will be k final clusters, but in the Map phase to get a good approximation of the data, O(k log n)
    // clusters are needed. Since n is the number of data points and not knowable until reading all the vectors,
    // provide a decent estimate.
    addOption(ESTIMATED_NUM_MAP_CLUSTERS, "km", "The estimated number of clusters to use for the "
        + "Map phase of the job when running StreamingKMeans. This should be around k * log(n), "
        + "where k is the final number of clusters and n is the total number of data points to "
        + "cluster.");

    addOption(ESTIMATED_DISTANCE_CUTOFF, "e", "The initial estimated distance cutoff between two "
        + "points for forming new clusters. If no value is given, it's estimated from the data set",
        String.valueOf(INVALID_DISTANCE_CUTOFF));

    // BallKMeans (reducer) options
    addOption(MAX_NUM_ITERATIONS, "mi", "The maximum number of iterations to run for the "
        + "BallKMeans algorithm used by the reducer. If no value is given, defaults to 10.", String.valueOf(10));

    addOption(TRIM_FRACTION, "tf", "The 'ball' aspect of ball k-means means that only the closest points "
        + "to the centroid will actually be used for updating. The fraction of the points to be used is those "
        + "points whose distance to the center is within trimFraction * distance to the closest other center. "
        + "If no value is given, defaults to 0.9.", String.valueOf(0.9));

    addFlag(RANDOM_INIT, "ri", "Whether to use k-means++ initialization or random initialization "
        + "of the seed centroids. Essentially, k-means++ provides better clusters, but takes longer, whereas random "
        + "initialization takes less time, but produces worse clusters, and tends to fail more often and needs "
        + "multiple runs to compare to k-means++. If set, uses the random initialization.");

    addFlag(IGNORE_WEIGHTS, "iw", "Whether to correct the weights of the centroids after the clustering is done. "
        + "The weights end up being wrong because of the trimFraction and possible train/test splits. In some cases, "
        + "especially in a pipeline, having an accurate count of the weights is useful. If set, ignores the final "
        + "weights");

    addOption(TEST_PROBABILITY, "testp", "A double value between 0 and 1 that represents the percentage of "
        + "points to be used for 'testing' different clustering runs in the final BallKMeans "
        + "step. If no value is given, defaults to 0.1", String.valueOf(0.1));

    addOption(NUM_BALLKMEANS_RUNS, "nbkm", "Number of BallKMeans runs to use at the end to try to cluster the "
        + "points. If no value is given, defaults to 4", String.valueOf(4));

    // Nearest neighbor search options
    // The distance measure used for computing the distance between two points. Generally, the
    // SquaredEuclideanDistance is used for clustering problems (it's equivalent to CosineDistance for normalized
    // vectors).
    // WARNING! You can use any metric but most of the literature is for the squared euclidean distance.
    addOption(DefaultOptionCreator.distanceMeasureOption().create());

    // The default searcher should be something more efficient that BruteSearch (ProjectionSearch, ...). See
    // o.a.m.math.neighborhood.*
    addOption(SEARCHER_CLASS_OPTION, "sc", "The type of searcher to be used when performing nearest "
        + "neighbor searches. Defaults to ProjectionSearch.", ProjectionSearch.class.getCanonicalName());

    // In the original paper, the authors used 1 projection vector.
    addOption(NUM_PROJECTIONS_OPTION, "np", "The number of projections considered in estimating the "
        + "distances between vectors. Only used when the distance measure requested is either "
        + "ProjectionSearch or FastProjectionSearch. If no value is given, defaults to 3.", String.valueOf(3));

    addOption(SEARCH_SIZE_OPTION, "s", "In more efficient searches (non BruteSearch), "
        + "not all distances are calculated for determining the nearest neighbors. The number of "
        + "elements whose distances from the query vector is actually computer is proportional to "
        + "searchSize. If no value is given, defaults to 1.", String.valueOf(2));

    addFlag(REDUCE_STREAMING_KMEANS, "rskm", "There might be too many intermediate clusters from the mapper "
        + "to fit into memory, so the reducer can run another pass of StreamingKMeans to collapse them down to a "
        + "fewer clusters");

    addOption(DefaultOptionCreator.methodOption().create());

    if (parseArguments(args) == null) {
      return -1;
    }
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }
    configureOptionsForWorkers();
    run(getConf(), getInputPath(), output);
    return 0;
  }

  private void configureOptionsForWorkers() throws ClassNotFoundException {
    log.info("Starting to configure options for workers");

    String method = getOption(DefaultOptionCreator.METHOD_OPTION);

    int numClusters = Integer.parseInt(getOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION));

    // StreamingKMeans
    int estimatedNumMapClusters = Integer.parseInt(getOption(ESTIMATED_NUM_MAP_CLUSTERS));
    float estimatedDistanceCutoff = Float.parseFloat(getOption(ESTIMATED_DISTANCE_CUTOFF));

    // BallKMeans
    int maxNumIterations = Integer.parseInt(getOption(MAX_NUM_ITERATIONS));
    float trimFraction = Float.parseFloat(getOption(TRIM_FRACTION));
    boolean randomInit = hasOption(RANDOM_INIT);
    boolean ignoreWeights = hasOption(IGNORE_WEIGHTS);
    float testProbability = Float.parseFloat(getOption(TEST_PROBABILITY));
    int numBallKMeansRuns = Integer.parseInt(getOption(NUM_BALLKMEANS_RUNS));

    // Nearest neighbor search
    String measureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    String searcherClass = getOption(SEARCHER_CLASS_OPTION);

    // Get more parameters depending on the kind of search class we're working with. BruteSearch
    // doesn't need anything else.
    // LocalitySensitiveHashSearch and ProjectionSearches need searchSize.
    // ProjectionSearches also need the number of projections.
    boolean getSearchSize = false;
    boolean getNumProjections = false;
    if (!searcherClass.equals(BruteSearch.class.getName())) {
      getSearchSize = true;
      getNumProjections = true;
    }

    // The search size to use. This is quite fuzzy and might end up not being configurable at all.
    int searchSize = 0;
    if (getSearchSize) {
      searchSize = Integer.parseInt(getOption(SEARCH_SIZE_OPTION));
    }

    // The number of projections to use. This is only useful in projection searches which
    // project the vectors on multiple basis vectors to get distance estimates that are faster to
    // calculate.
    int numProjections = 0;
    if (getNumProjections) {
      numProjections = Integer.parseInt(getOption(NUM_PROJECTIONS_OPTION));
    }

    boolean reduceStreamingKMeans = hasOption(REDUCE_STREAMING_KMEANS);

    configureOptionsForWorkers(getConf(), numClusters,
        /* StreamingKMeans */
        estimatedNumMapClusters,  estimatedDistanceCutoff,
        /* BallKMeans */
        maxNumIterations, trimFraction, randomInit, ignoreWeights, testProbability, numBallKMeansRuns,
        /* Searcher */
        measureClass, searcherClass,  searchSize, numProjections,
        method,
        reduceStreamingKMeans);
  }

  /**
   * Checks the parameters for a StreamingKMeans job and prepares a Configuration with them.
   *
   * @param conf the Configuration to populate
   * @param numClusters k, the number of clusters at the end
   * @param estimatedNumMapClusters O(k log n), the number of clusters requested from each mapper
   * @param estimatedDistanceCutoff an estimate of the minimum distance that separates two clusters (can be smaller and
   *                                will be increased dynamically)
   * @param maxNumIterations the maximum number of iterations of BallKMeans
   * @param trimFraction the fraction of the points to be considered in updating a ball k-means
   * @param randomInit whether to initialize the ball k-means seeds randomly
   * @param ignoreWeights whether to ignore the invalid final ball k-means weights
   * @param testProbability the percentage of vectors assigned to the test set for selecting the best final centers
   * @param numBallKMeansRuns the number of BallKMeans runs in the reducer that determine the centroids to return
   *                          (clusters are computed for the training set and the error is computed on the test set)
   * @param measureClass string, name of the distance measure class; theory works for Euclidean-like distances
   * @param searcherClass string, name of the searcher that will be used for nearest neighbor search
   * @param searchSize the number of closest neighbors to look at for selecting the closest one in approximate nearest
   *                   neighbor searches
   * @param numProjections the number of projected vectors to use for faster searching (only useful for ProjectionSearch
   *                       or FastProjectionSearch); @see org.apache.mahout.math.neighborhood.ProjectionSearch
   */
  public static void configureOptionsForWorkers(Configuration conf,
                                                int numClusters,
                                                /* StreamingKMeans */
                                                int estimatedNumMapClusters, float estimatedDistanceCutoff,
                                                /* BallKMeans */
                                                int maxNumIterations, float trimFraction, boolean randomInit,
                                                boolean ignoreWeights, float testProbability, int numBallKMeansRuns,
                                                /* Searcher */
                                                String measureClass, String searcherClass,
                                                int searchSize, int numProjections,
                                                String method,
                                                boolean reduceStreamingKMeans) throws ClassNotFoundException {
    // Checking preconditions for the parameters.
    Preconditions.checkArgument(numClusters > 0, 
        "Invalid number of clusters requested: " + numClusters + ". Must be: numClusters > 0!");

    // StreamingKMeans
    Preconditions.checkArgument(estimatedNumMapClusters > numClusters, "Invalid number of estimated map "
        + "clusters; There must be more than the final number of clusters (k log n vs k)");
    Preconditions.checkArgument(estimatedDistanceCutoff == INVALID_DISTANCE_CUTOFF || estimatedDistanceCutoff > 0,
        "estimatedDistanceCutoff must be equal to -1 or must be greater then 0!");

    // BallKMeans
    Preconditions.checkArgument(maxNumIterations > 0, "Must have at least one BallKMeans iteration");
    Preconditions.checkArgument(trimFraction > 0, "trimFraction must be positive");
    Preconditions.checkArgument(testProbability >= 0 && testProbability < 1, "test probability is not in the "
        + "interval [0, 1)");
    Preconditions.checkArgument(numBallKMeansRuns > 0, "numBallKMeans cannot be negative");

    // Searcher
    if (!searcherClass.contains("Brute")) {
      // These tests only make sense when a relevant searcher is being used.
      Preconditions.checkArgument(searchSize > 0, "Invalid searchSize. Must be positive.");
      if (searcherClass.contains("Projection")) {
        Preconditions.checkArgument(numProjections > 0, "Invalid numProjections. Must be positive");
      }
    }

    // Setting the parameters in the Configuration.
    conf.setInt(DefaultOptionCreator.NUM_CLUSTERS_OPTION, numClusters);
    /* StreamingKMeans */
    conf.setInt(ESTIMATED_NUM_MAP_CLUSTERS, estimatedNumMapClusters);
    if (estimatedDistanceCutoff != INVALID_DISTANCE_CUTOFF) {
      conf.setFloat(ESTIMATED_DISTANCE_CUTOFF, estimatedDistanceCutoff);
    }
    /* BallKMeans */
    conf.setInt(MAX_NUM_ITERATIONS, maxNumIterations);
    conf.setFloat(TRIM_FRACTION, trimFraction);
    conf.setBoolean(RANDOM_INIT, randomInit);
    conf.setBoolean(IGNORE_WEIGHTS, ignoreWeights);
    conf.setFloat(TEST_PROBABILITY, testProbability);
    conf.setInt(NUM_BALLKMEANS_RUNS, numBallKMeansRuns);
    /* Searcher */
    // Checks if the measureClass is available, throws exception otherwise.
    Class.forName(measureClass);
    conf.set(DefaultOptionCreator.DISTANCE_MEASURE_OPTION, measureClass);
    // Checks if the searcherClass is available, throws exception otherwise.
    Class.forName(searcherClass);
    conf.set(SEARCHER_CLASS_OPTION, searcherClass);
    conf.setInt(SEARCH_SIZE_OPTION, searchSize);
    conf.setInt(NUM_PROJECTIONS_OPTION, numProjections);
    conf.set(DefaultOptionCreator.METHOD_OPTION, method);

    conf.setBoolean(REDUCE_STREAMING_KMEANS, reduceStreamingKMeans);

    log.info("Parameters are: [k] numClusters {}; "
        + "[SKM] estimatedNumMapClusters {}; estimatedDistanceCutoff {} "
        + "[BKM] maxNumIterations {}; trimFraction {}; randomInit {}; ignoreWeights {}; "
        + "testProbability {}; numBallKMeansRuns {}; "
        + "[S] measureClass {}; searcherClass {}; searcherSize {}; numProjections {}; "
        + "method {}; reduceStreamingKMeans {}", numClusters, estimatedNumMapClusters, estimatedDistanceCutoff,
        maxNumIterations, trimFraction, randomInit, ignoreWeights, testProbability, numBallKMeansRuns,
        measureClass, searcherClass, searchSize, numProjections, method, reduceStreamingKMeans);
  }

  /**
   * Iterate over the input vectors to produce clusters and, if requested, use the results of the final iteration to
   * cluster the input vectors.
   *
   * @param input the directory pathname for input points.
   * @param output the directory pathname for output points.
   * @return 0 on success, -1 on failure.
   */
  public static int run(Configuration conf, Path input, Path output)
      throws IOException, InterruptedException, ClassNotFoundException, ExecutionException {
    log.info("Starting StreamingKMeans clustering for vectors in {}; results are output to {}",
        input.toString(), output.toString());

    if (conf.get(DefaultOptionCreator.METHOD_OPTION,
        DefaultOptionCreator.MAPREDUCE_METHOD).equals(DefaultOptionCreator.SEQUENTIAL_METHOD)) {
      return runSequentially(conf, input, output);
    } else {
      return runMapReduce(conf, input, output);
    }
  }

  private static int runSequentially(Configuration conf, Path input, Path output)
    throws IOException, ExecutionException, InterruptedException {
    long start = System.currentTimeMillis();
    // Run StreamingKMeans step in parallel by spawning 1 thread per input path to process.
    ExecutorService pool = Executors.newCachedThreadPool();
    List<Future<Iterable<Centroid>>> intermediateCentroidFutures = Lists.newArrayList();
    for (FileStatus status : HadoopUtil.listStatus(FileSystem.get(conf), input, PathFilters.logsCRCFilter())) {
      intermediateCentroidFutures.add(pool.submit(new StreamingKMeansThread(status.getPath(), conf)));
    }
    log.info("Finished running Mappers");
    // Merge the resulting "mapper" centroids.
    List<Centroid> intermediateCentroids = Lists.newArrayList();
    for (Future<Iterable<Centroid>> futureIterable : intermediateCentroidFutures) {
      for (Centroid centroid : futureIterable.get()) {
        intermediateCentroids.add(centroid);
      }
    }
    pool.shutdown();
    pool.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS);
    log.info("Finished StreamingKMeans");
    SequenceFile.Writer writer = SequenceFile.createWriter(FileSystem.get(conf), conf, new Path(output, "part-r-00000"), IntWritable.class,
        CentroidWritable.class);
    int numCentroids = 0;
    // Run BallKMeans on the intermediate centroids.
    for (Vector finalVector : StreamingKMeansReducer.getBestCentroids(intermediateCentroids, conf)) {
      Centroid finalCentroid = (Centroid)finalVector;
      writer.append(new IntWritable(numCentroids++), new CentroidWritable(finalCentroid));
    }
    writer.close();
    long end = System.currentTimeMillis();
    log.info("Finished BallKMeans. Took {}.", (end - start) / 1000.0);
    return 0;
  }

  public static int runMapReduce(Configuration conf, Path input, Path output)
    throws IOException, ClassNotFoundException, InterruptedException {
    // Prepare Job for submission.
    Job job = HadoopUtil.prepareJob(input, output, SequenceFileInputFormat.class,
        StreamingKMeansMapper.class, IntWritable.class, CentroidWritable.class,
        StreamingKMeansReducer.class, IntWritable.class, CentroidWritable.class, SequenceFileOutputFormat.class,
        conf);
    job.setJobName(HadoopUtil.getCustomJobName(StreamingKMeansDriver.class.getSimpleName(), job,
        StreamingKMeansMapper.class, StreamingKMeansReducer.class));

    // There is only one reducer so that the intermediate centroids get collected on one
    // machine and are clustered in memory to get the right number of clusters.
    job.setNumReduceTasks(1);

    // Set the JAR (so that the required libraries are available) and run.
    job.setJarByClass(StreamingKMeansDriver.class);

    // Run job!
    long start = System.currentTimeMillis();
    if (!job.waitForCompletion(true)) {
      return -1;
    }
    long end = System.currentTimeMillis();

    log.info("StreamingKMeans clustering complete. Results are in {}. Took {} ms", output.toString(), end - start);
    return 0;
  }

  /**
   * Constructor to be used by the ToolRunner.
   */
  private StreamingKMeansDriver() {}

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new StreamingKMeansDriver(), args);
  }
}
