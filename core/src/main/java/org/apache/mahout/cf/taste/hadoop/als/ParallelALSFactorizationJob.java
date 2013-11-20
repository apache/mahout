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

package org.apache.mahout.cf.taste.hadoop.als;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.google.common.base.Preconditions;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.map.MultithreadedMapper;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.mapreduce.MergeVectorsCombiner;
import org.apache.mahout.common.mapreduce.MergeVectorsReducer;
import org.apache.mahout.common.mapreduce.TransposeMapper;
import org.apache.mahout.common.mapreduce.VectorSumCombiner;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.Vectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>MapReduce implementation of the two factorization algorithms described in
 *
 * <p>"Large-scale Parallel Collaborative Filtering for the Netﬂix Prize" available at
 * http://www.hpl.hp.com/personal/Robert_Schreiber/papers/2008%20AAIM%20Netflix/netflix_aaim08(submitted).pdf.</p>
 *
 * "<p>Collaborative Filtering for Implicit Feedback Datasets" available at
 * http://research.yahoo.com/pub/2433</p>
 *
 * </p>
 * <p>Command line arguments specific to this class are:</p>
 *
 * <ol>
 * <li>--input (path): Directory containing one or more text files with the dataset</li>
 * <li>--output (path): path where output should go</li>
 * <li>--lambda (double): regularization parameter to avoid overfitting</li>
 * <li>--userFeatures (path): path to the user feature matrix</li>
 * <li>--itemFeatures (path): path to the item feature matrix</li>
 * <li>--numThreadsPerSolver (int): threads to use per solver mapper, (default: 1)</li>
 * </ol>
 */
public class ParallelALSFactorizationJob extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(ParallelALSFactorizationJob.class);

  static final String NUM_FEATURES = ParallelALSFactorizationJob.class.getName() + ".numFeatures";
  static final String LAMBDA = ParallelALSFactorizationJob.class.getName() + ".lambda";
  static final String ALPHA = ParallelALSFactorizationJob.class.getName() + ".alpha";
  static final String NUM_ENTITIES = ParallelALSFactorizationJob.class.getName() + ".numEntities";

  static final String USES_LONG_IDS = ParallelALSFactorizationJob.class.getName() + ".usesLongIDs";
  static final String TOKEN_POS = ParallelALSFactorizationJob.class.getName() + ".tokenPos";

  private boolean implicitFeedback;
  private int numIterations;
  private int numFeatures;
  private double lambda;
  private double alpha;
  private int numThreadsPerSolver;
  private boolean usesLongIDs;

  private int numItems;
  private int numUsers;

  enum Stats { NUM_USERS }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new ParallelALSFactorizationJob(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption("lambda", null, "regularization parameter", true);
    addOption("implicitFeedback", null, "data consists of implicit feedback?", String.valueOf(false));
    addOption("alpha", null, "confidence parameter (only used on implicit feedback)", String.valueOf(40));
    addOption("numFeatures", null, "dimension of the feature space", true);
    addOption("numIterations", null, "number of iterations", true);
    addOption("numThreadsPerSolver", null, "threads per solver mapper", String.valueOf(1));
    addOption("usesLongIDs", null, "input contains long IDs that need to be translated");

    Map<String,List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    numFeatures = Integer.parseInt(getOption("numFeatures"));
    numIterations = Integer.parseInt(getOption("numIterations"));
    lambda = Double.parseDouble(getOption("lambda"));
    alpha = Double.parseDouble(getOption("alpha"));
    implicitFeedback = Boolean.parseBoolean(getOption("implicitFeedback"));

    numThreadsPerSolver = Integer.parseInt(getOption("numThreadsPerSolver"));
    usesLongIDs = Boolean.parseBoolean(getOption("usesLongIDs", String.valueOf(false)));

    /*
    * compute the factorization A = U M'
    *
    * where A (users x items) is the matrix of known ratings
    *           U (users x features) is the representation of users in the feature space
    *           M (items x features) is the representation of items in the feature space
    */

    if (usesLongIDs) {
      Job mapUsers = prepareJob(getInputPath(), getOutputPath("userIDIndex"), TextInputFormat.class,
          MapLongIDsMapper.class, VarIntWritable.class, VarLongWritable.class, IDMapReducer.class,
          VarIntWritable.class, VarLongWritable.class, SequenceFileOutputFormat.class);
      mapUsers.getConfiguration().set(TOKEN_POS, String.valueOf(TasteHadoopUtils.USER_ID_POS));
      mapUsers.waitForCompletion(true);

      Job mapItems = prepareJob(getInputPath(), getOutputPath("itemIDIndex"), TextInputFormat.class,
          MapLongIDsMapper.class, VarIntWritable.class, VarLongWritable.class, IDMapReducer.class,
          VarIntWritable.class, VarLongWritable.class, SequenceFileOutputFormat.class);
      mapItems.getConfiguration().set(TOKEN_POS, String.valueOf(TasteHadoopUtils.ITEM_ID_POS));
      mapItems.waitForCompletion(true);
    }

   /* create A' */
    Job itemRatings = prepareJob(getInputPath(), pathToItemRatings(),
        TextInputFormat.class, ItemRatingVectorsMapper.class, IntWritable.class,
        VectorWritable.class, VectorSumReducer.class, IntWritable.class,
        VectorWritable.class, SequenceFileOutputFormat.class);
    itemRatings.setCombinerClass(VectorSumCombiner.class);
    itemRatings.getConfiguration().set(USES_LONG_IDS, String.valueOf(usesLongIDs));
    boolean succeeded = itemRatings.waitForCompletion(true);
    if (!succeeded) {
      return -1;
    }

    /* create A */
    Job userRatings = prepareJob(pathToItemRatings(), pathToUserRatings(),
        TransposeMapper.class, IntWritable.class, VectorWritable.class, MergeUserVectorsReducer.class,
        IntWritable.class, VectorWritable.class);
    userRatings.setCombinerClass(MergeVectorsCombiner.class);
    succeeded = userRatings.waitForCompletion(true);
    if (!succeeded) {
      return -1;
    }

    //TODO this could be fiddled into one of the upper jobs
    Job averageItemRatings = prepareJob(pathToItemRatings(), getTempPath("averageRatings"),
        AverageRatingMapper.class, IntWritable.class, VectorWritable.class, MergeVectorsReducer.class,
        IntWritable.class, VectorWritable.class);
    averageItemRatings.setCombinerClass(MergeVectorsCombiner.class);
    succeeded = averageItemRatings.waitForCompletion(true);
    if (!succeeded) {
      return -1;
    }

    Vector averageRatings = ALS.readFirstRow(getTempPath("averageRatings"), getConf());

    numItems = averageRatings.getNumNondefaultElements();
    numUsers = (int) userRatings.getCounters().findCounter(Stats.NUM_USERS).getValue();

    log.info("Found {} users and {} items", numUsers, numItems);

    /* create an initial M */
    initializeM(averageRatings);

    for (int currentIteration = 0; currentIteration < numIterations; currentIteration++) {
      /* broadcast M, read A row-wise, recompute U row-wise */
      log.info("Recomputing U (iteration {}/{})", currentIteration, numIterations);
      runSolver(pathToUserRatings(), pathToU(currentIteration), pathToM(currentIteration - 1), currentIteration, "U",
                numItems);
      /* broadcast U, read A' row-wise, recompute M row-wise */
      log.info("Recomputing M (iteration {}/{})", currentIteration, numIterations);
      runSolver(pathToItemRatings(), pathToM(currentIteration), pathToU(currentIteration), currentIteration, "M",
                numUsers);
    }

    return 0;
  }

  private void initializeM(Vector averageRatings) throws IOException {
    Random random = RandomUtils.getRandom();

    FileSystem fs = FileSystem.get(pathToM(-1).toUri(), getConf());
    SequenceFile.Writer writer = null;
    try {
      writer = new SequenceFile.Writer(fs, getConf(), new Path(pathToM(-1), "part-m-00000"), IntWritable.class,
          VectorWritable.class);

      IntWritable index = new IntWritable();
      VectorWritable featureVector = new VectorWritable();

      for (Vector.Element e : averageRatings.nonZeroes()) {
        Vector row = new DenseVector(numFeatures);
        row.setQuick(0, e.get());
        for (int m = 1; m < numFeatures; m++) {
          row.setQuick(m, random.nextDouble());
        }
        index.set(e.index());
        featureVector.set(row);
        writer.append(index, featureVector);
      }
    } finally {
      Closeables.close(writer, false);
    }
  }

  static class VectorSumReducer
      extends Reducer<WritableComparable<?>, VectorWritable, WritableComparable<?>, VectorWritable> {

    private final VectorWritable result = new VectorWritable();

    @Override
    protected void reduce(WritableComparable<?> key, Iterable<VectorWritable> values, Context ctx)
      throws IOException, InterruptedException {
      Vector sum = Vectors.sum(values.iterator());
      result.set(new SequentialAccessSparseVector(sum));
      ctx.write(key, result);
    }
  }

  static class MergeUserVectorsReducer extends
      Reducer<WritableComparable<?>,VectorWritable,WritableComparable<?>,VectorWritable> {

    private final VectorWritable result = new VectorWritable();

    @Override
    public void reduce(WritableComparable<?> key, Iterable<VectorWritable> vectors, Context ctx)
      throws IOException, InterruptedException {
      Vector merged = VectorWritable.merge(vectors.iterator()).get();
      result.set(new SequentialAccessSparseVector(merged));
      ctx.write(key, result);
      ctx.getCounter(Stats.NUM_USERS).increment(1);
    }
  }

  static class ItemRatingVectorsMapper extends Mapper<LongWritable,Text,IntWritable,VectorWritable> {

    private final IntWritable itemIDWritable = new IntWritable();
    private final VectorWritable ratingsWritable = new VectorWritable(true);
    private final Vector ratings = new RandomAccessSparseVector(Integer.MAX_VALUE, 1);

    private boolean usesLongIDs;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      usesLongIDs = ctx.getConfiguration().getBoolean(USES_LONG_IDS, false);
    }

    @Override
    protected void map(LongWritable offset, Text line, Context ctx) throws IOException, InterruptedException {
      String[] tokens = TasteHadoopUtils.splitPrefTokens(line.toString());
      int userID = TasteHadoopUtils.readID(tokens[TasteHadoopUtils.USER_ID_POS], usesLongIDs);
      int itemID = TasteHadoopUtils.readID(tokens[TasteHadoopUtils.ITEM_ID_POS], usesLongIDs);
      float rating = Float.parseFloat(tokens[2]);

      ratings.setQuick(userID, rating);

      itemIDWritable.set(itemID);
      ratingsWritable.set(ratings);

      ctx.write(itemIDWritable, ratingsWritable);

      // prepare instance for reuse
      ratings.setQuick(userID, 0.0d);
    }
  }

  private void runSolver(Path ratings, Path output, Path pathToUorM, int currentIteration, String matrixName,
                         int numEntities) throws ClassNotFoundException, IOException, InterruptedException {

    // necessary for local execution in the same JVM only
    SharingMapper.reset();

    Class<? extends Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable>> solverMapperClassInternal;
    String name;

    if (implicitFeedback) {
      solverMapperClassInternal = SolveImplicitFeedbackMapper.class;
      name = "Recompute " + matrixName + ", iteration (" + currentIteration + '/' + numIterations + "), "
          + '(' + numThreadsPerSolver + " threads, " + numFeatures + " features, implicit feedback)";
    } else {
      solverMapperClassInternal = SolveExplicitFeedbackMapper.class;
      name = "Recompute " + matrixName + ", iteration (" + currentIteration + '/' + numIterations + "), "
          + '(' + numThreadsPerSolver + " threads, " + numFeatures + " features, explicit feedback)";
    }

    Job solverForUorI = prepareJob(ratings, output, SequenceFileInputFormat.class, MultithreadedSharingMapper.class,
        IntWritable.class, VectorWritable.class, SequenceFileOutputFormat.class, name);
    Configuration solverConf = solverForUorI.getConfiguration();
    solverConf.set(LAMBDA, String.valueOf(lambda));
    solverConf.set(ALPHA, String.valueOf(alpha));
    solverConf.setInt(NUM_FEATURES, numFeatures);
    solverConf.set(NUM_ENTITIES, String.valueOf(numEntities));

    FileSystem fs = FileSystem.get(pathToUorM.toUri(), solverConf);
    FileStatus[] parts = fs.listStatus(pathToUorM, PathFilters.partFilter());
    for (FileStatus part : parts) {
      if (log.isDebugEnabled()) {
        log.debug("Adding {} to distributed cache", part.getPath().toString());
      }
      DistributedCache.addCacheFile(part.getPath().toUri(), solverConf);
    }

    MultithreadedMapper.setMapperClass(solverForUorI, solverMapperClassInternal);
    MultithreadedMapper.setNumberOfThreads(solverForUorI, numThreadsPerSolver);

    boolean succeeded = solverForUorI.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }
  }

  static class AverageRatingMapper extends Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {

    private final IntWritable firstIndex = new IntWritable(0);
    private final Vector featureVector = new RandomAccessSparseVector(Integer.MAX_VALUE, 1);
    private final VectorWritable featureVectorWritable = new VectorWritable();

    @Override
    protected void map(IntWritable r, VectorWritable v, Context ctx) throws IOException, InterruptedException {
      RunningAverage avg = new FullRunningAverage();
      for (Vector.Element e : v.get().nonZeroes()) {
        avg.addDatum(e.get());
      }

      featureVector.setQuick(r.get(), avg.getAverage());
      featureVectorWritable.set(featureVector);
      ctx.write(firstIndex, featureVectorWritable);

      // prepare instance for reuse
      featureVector.setQuick(r.get(), 0.0d);
    }
  }

  static class MapLongIDsMapper extends Mapper<LongWritable,Text,VarIntWritable,VarLongWritable> {

    private int tokenPos;
    private final VarIntWritable index = new VarIntWritable();
    private final VarLongWritable idWritable = new VarLongWritable();

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      tokenPos = ctx.getConfiguration().getInt(TOKEN_POS, -1);
      Preconditions.checkState(tokenPos >= 0);
    }

    @Override
    protected void map(LongWritable key, Text line, Context ctx) throws IOException, InterruptedException {
      String[] tokens = TasteHadoopUtils.splitPrefTokens(line.toString());

      long id = Long.parseLong(tokens[tokenPos]);

      index.set(TasteHadoopUtils.idToIndex(id));
      idWritable.set(id);
      ctx.write(index, idWritable);
    }
  }

  static class IDMapReducer extends Reducer<VarIntWritable,VarLongWritable,VarIntWritable,VarLongWritable> {
    @Override
    protected void reduce(VarIntWritable index, Iterable<VarLongWritable> ids, Context ctx)
      throws IOException, InterruptedException {
      ctx.write(index, ids.iterator().next());
    }
  }

  private Path pathToM(int iteration) {
    return iteration == numIterations - 1 ? getOutputPath("M") : getTempPath("M-" + iteration);
  }

  private Path pathToU(int iteration) {
    return iteration == numIterations - 1 ? getOutputPath("U") : getTempPath("U-" + iteration);
  }

  private Path pathToItemRatings() {
    return getTempPath("itemRatings");
  }

  private Path pathToUserRatings() {
    return getOutputPath("userRatings");
  }
}
