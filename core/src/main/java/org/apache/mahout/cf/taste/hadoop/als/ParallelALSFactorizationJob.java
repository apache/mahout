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

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.mapreduce.MergeVectorsCombiner;
import org.apache.mahout.common.mapreduce.MergeVectorsReducer;
import org.apache.mahout.common.mapreduce.TransposeMapper;
import org.apache.mahout.common.mapreduce.VectorSumReducer;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.als.AlternatingLeastSquaresSolver;
import org.apache.mahout.math.als.ImplicitFeedbackAlternatingLeastSquaresSolver;
import org.apache.mahout.math.map.OpenIntObjectHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

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
 * </ol>
 */
public class ParallelALSFactorizationJob extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(ParallelALSFactorizationJob.class);

  static final String NUM_FEATURES = ParallelALSFactorizationJob.class.getName() + ".numFeatures";
  static final String LAMBDA = ParallelALSFactorizationJob.class.getName() + ".lambda";
  static final String ALPHA = ParallelALSFactorizationJob.class.getName() + ".alpha";
  static final String FEATURE_MATRIX = ParallelALSFactorizationJob.class.getName() + ".featureMatrix";

  private boolean implicitFeedback;
  private int numIterations;
  private int numFeatures;
  private double lambda;
  private double alpha;

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

    Map<String,String> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    numFeatures = Integer.parseInt(parsedArgs.get("--numFeatures"));
    numIterations = Integer.parseInt(parsedArgs.get("--numIterations"));
    lambda = Double.parseDouble(parsedArgs.get("--lambda"));
    alpha = Double.parseDouble(parsedArgs.get("--alpha"));
    implicitFeedback = Boolean.parseBoolean(parsedArgs.get("--implicitFeedback"));

    /*
        * compute the factorization A = U M'
        *
        * where A (users x items) is the matrix of known ratings
        *           U (users x features) is the representation of users in the feature space
        *           M (items x features) is the representation of items in the feature space
        */

   /* create A' */
    Job itemRatings = prepareJob(getInputPath(), pathToItemRatings(),
        TextInputFormat.class, ItemRatingVectorsMapper.class, IntWritable.class,
        VectorWritable.class, VectorSumReducer.class, IntWritable.class,
        VectorWritable.class, SequenceFileOutputFormat.class);
    itemRatings.setCombinerClass(VectorSumReducer.class);
    itemRatings.waitForCompletion(true);

    /* create A */
    Job userRatings = prepareJob(pathToItemRatings(), pathToUserRatings(),
        TransposeMapper.class, IntWritable.class, VectorWritable.class, MergeVectorsReducer.class, IntWritable.class,
        VectorWritable.class);
    userRatings.setCombinerClass(MergeVectorsCombiner.class);
    userRatings.waitForCompletion(true);

    //TODO this could be fiddled into one of the upper jobs
    Job averageItemRatings = prepareJob(pathToItemRatings(), getTempPath("averageRatings"),
        AverageRatingMapper.class, IntWritable.class, VectorWritable.class, MergeVectorsReducer.class,
        IntWritable.class, VectorWritable.class);
    averageItemRatings.setCombinerClass(MergeVectorsCombiner.class);
    averageItemRatings.waitForCompletion(true);

    Vector averageRatings = ALSUtils.readFirstRow(getTempPath("averageRatings"), getConf());

    /* create an initial M */
    initializeM(averageRatings);

    for (int currentIteration = 0; currentIteration < numIterations; currentIteration++) {
      /* broadcast M, read A row-wise, recompute U row-wise */
      log.info("Recomputing U (iteration {}/{})", currentIteration, numIterations);
      runSolver(pathToUserRatings(), pathToU(currentIteration), pathToM(currentIteration - 1));
      /* broadcast U, read A' row-wise, recompute M row-wise */
      log.info("Recomputing M (iteration {}/{})", currentIteration, numIterations);
      runSolver(pathToItemRatings(), pathToM(currentIteration), pathToU(currentIteration));
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

      Iterator<Vector.Element> averages = averageRatings.iterateNonZero();
      while (averages.hasNext()) {
        Vector.Element e = averages.next();
        Vector row = new DenseVector(numFeatures);
        row.setQuick(0, e.get());
        for (int m = 1; m < numFeatures; m++) {
          row.setQuick(m, random.nextDouble());
        }
        writer.append(new IntWritable(e.index()), new VectorWritable(row));
      }
    } finally {
      Closeables.closeQuietly(writer);
    }
  }

  static class ItemRatingVectorsMapper extends Mapper<LongWritable,Text,IntWritable,VectorWritable> {
    @Override
    protected void map(LongWritable offset, Text line, Context ctx) throws IOException, InterruptedException {
      String[] tokens = TasteHadoopUtils.splitPrefTokens(line.toString());
      int userID = Integer.parseInt(tokens[0]);
      int itemID = Integer.parseInt(tokens[1]);
      float rating = Float.parseFloat(tokens[2]);

      Vector ratings = new RandomAccessSparseVector(Integer.MAX_VALUE, 1);
      ratings.set(userID, rating);

      ctx.write(new IntWritable(itemID), new VectorWritable(ratings, true));
    }
  }

  private void runSolver(Path ratings, Path output, Path pathToUorI)
      throws ClassNotFoundException, IOException, InterruptedException {

    Class<? extends Mapper> solverMapper = implicitFeedback ?
        SolveImplicitFeedbackMapper.class : SolveExplicitFeedbackMapper.class;

    Job solverForUorI = prepareJob(ratings, output, SequenceFileInputFormat.class, solverMapper, IntWritable.class,
        VectorWritable.class, SequenceFileOutputFormat.class);
    Configuration solverConf = solverForUorI.getConfiguration();
    solverConf.set(LAMBDA, String.valueOf(lambda));
    solverConf.set(ALPHA, String.valueOf(alpha));
    solverConf.setInt(NUM_FEATURES, numFeatures);
    solverConf.set(FEATURE_MATRIX, pathToUorI.toString());
    solverForUorI.waitForCompletion(true);
  }

  static class SolveExplicitFeedbackMapper extends Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {

    private double lambda;
    private int numFeatures;

    private OpenIntObjectHashMap<Vector> UorM;

    private AlternatingLeastSquaresSolver solver;

    @Override
    protected void setup(Mapper.Context ctx) throws IOException, InterruptedException {
      lambda = Double.parseDouble(ctx.getConfiguration().get(LAMBDA));
      numFeatures = ctx.getConfiguration().getInt(NUM_FEATURES, -1);
      solver = new AlternatingLeastSquaresSolver();

      Path UOrIPath = new Path(ctx.getConfiguration().get(FEATURE_MATRIX));

      UorM = ALSUtils.readMatrixByRows(UOrIPath, ctx.getConfiguration());
      Preconditions.checkArgument(numFeatures > 0, "numFeatures was not set correctly!");
    }

    @Override
    protected void map(IntWritable userOrItemID, VectorWritable ratingsWritable, Context ctx)
        throws IOException, InterruptedException {
      Vector ratings = new SequentialAccessSparseVector(ratingsWritable.get());
      List<Vector> featureVectors = Lists.newArrayList();
      Iterator<Vector.Element> interactions = ratings.iterateNonZero();
      while (interactions.hasNext()) {
        int index = interactions.next().index();
        featureVectors.add(UorM.get(index));
      }

      Vector uiOrmj = solver.solve(featureVectors, ratings, lambda, numFeatures);

      ctx.write(userOrItemID, new VectorWritable(uiOrmj));
    }
  }

  static class SolveImplicitFeedbackMapper extends Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {

    private ImplicitFeedbackAlternatingLeastSquaresSolver solver;

    @Override
    protected void setup(Mapper.Context ctx) throws IOException, InterruptedException {
      double lambda = Double.parseDouble(ctx.getConfiguration().get(LAMBDA));
      double alpha = Double.parseDouble(ctx.getConfiguration().get(ALPHA));
      int numFeatures = ctx.getConfiguration().getInt(NUM_FEATURES, -1);

      Path YPath = new Path(ctx.getConfiguration().get(FEATURE_MATRIX));
      OpenIntObjectHashMap<Vector> Y = ALSUtils.readMatrixByRows(YPath, ctx.getConfiguration());

      solver = new ImplicitFeedbackAlternatingLeastSquaresSolver(numFeatures, lambda, alpha, Y);

      Preconditions.checkArgument(numFeatures > 0, "numFeatures was not set correctly!");
    }

    @Override
    protected void map(IntWritable userOrItemID, VectorWritable ratingsWritable, Context ctx)
        throws IOException, InterruptedException {
      Vector ratings = new SequentialAccessSparseVector(ratingsWritable.get());

      Vector uiOrmj = solver.solve(ratings);

      ctx.write(userOrItemID, new VectorWritable(uiOrmj));
    }
  }

  static class AverageRatingMapper extends Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {
    @Override
    protected void map(IntWritable r, VectorWritable v, Context ctx) throws IOException, InterruptedException {
      RunningAverage avg = new FullRunningAverage();
      Iterator<Vector.Element> elements = v.get().iterateNonZero();
      while (elements.hasNext()) {
        avg.addDatum(elements.next().get());
      }
      Vector vector = new RandomAccessSparseVector(Integer.MAX_VALUE, 1);
      vector.setQuick(r.get(), avg.getAverage());
      ctx.write(new IntWritable(0), new VectorWritable(vector));
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
