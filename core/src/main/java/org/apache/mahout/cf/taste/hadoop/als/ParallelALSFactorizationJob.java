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
import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.als.AlternateLeastSquaresSolver;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * <p>MapReduce implementation of the factorization algorithm described in
 * "Large-scale Parallel Collaborative Filtering for the Netﬂix Prize"
 * available at
 * http://www.hpl.hp.com/personal/Robert_Schreiber/papers/2008%20AAIM%20Netflix/netflix_aaim08(submitted).pdf.</p>
 *
 * <p>Implements a parallel algorithm that uses "Alternating-Least-Squares with Weighted-λ-Regularization"
 * to factorize the preference-matrix </p>
 *
 * <p>Command line arguments specific to this class are:</p>
 *
 * <ol>
 * <li>--input (path): Directory containing one or more text files with the dataset</li>
 * <li>--output (path): path where output should go</li>
 * <li>--lambda (double): regularization parameter to avoid overfitting</li>
 * <li>--numFeatures (int): number of features to use for decomposition </li>
* <li>--numIterations (int): number of iterations to run</li>
 * </ol>
 */
public class ParallelALSFactorizationJob extends AbstractJob {

  static final String NUM_FEATURES = ParallelALSFactorizationJob.class.getName() + ".numFeatures";
  static final String LAMBDA = ParallelALSFactorizationJob.class.getName() + ".lambda";
  static final String MAP_TRANSPOSED = ParallelALSFactorizationJob.class.getName() + ".mapTransposed";

  static final String STEP_ONE = "fixMcomputeU";
  static final String STEP_TWO = "fixUcomputeM";

  private String tempDir;

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new ParallelALSFactorizationJob(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption("lambda", "l", "regularization parameter", true);
    addOption("numFeatures", "f", "dimension of the feature space", true);
    addOption("numIterations", "i", "number of iterations", true);

    Map<String,String> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    int numFeatures = Integer.parseInt(parsedArgs.get("--numFeatures"));
    int numIterations = Integer.parseInt(parsedArgs.get("--numIterations"));
    double lambda = Double.parseDouble(parsedArgs.get("--lambda"));
    tempDir = parsedArgs.get("--tempDir");

    Job itemRatings = prepareJob(getInputPath(), pathToItemRatings(),
        TextInputFormat.class, PrefsToRatingsMapper.class, VarIntWritable.class,
        FeatureVectorWithRatingWritable.class, Reducer.class, VarIntWritable.class,
        FeatureVectorWithRatingWritable.class, SequenceFileOutputFormat.class);
    itemRatings.waitForCompletion(true);
    
    Job userRatings = prepareJob(getInputPath(), pathToUserRatings(),
        TextInputFormat.class, PrefsToRatingsMapper.class, VarIntWritable.class,
        FeatureVectorWithRatingWritable.class, Reducer.class, VarIntWritable.class,
        FeatureVectorWithRatingWritable.class, SequenceFileOutputFormat.class);
    userRatings.getConfiguration().setBoolean(MAP_TRANSPOSED, Boolean.TRUE);
    userRatings.waitForCompletion(true);

    Job initializeM = prepareJob(getInputPath(), pathToM(-1), TextInputFormat.class, ItemIDRatingMapper.class,
        VarLongWritable.class, FloatWritable.class, InitializeMReducer.class, VarIntWritable.class,
        FeatureVectorWithRatingWritable.class, SequenceFileOutputFormat.class);
    initializeM.getConfiguration().setInt(NUM_FEATURES, numFeatures);
    initializeM.waitForCompletion(true);

    for (int n = 0; n < numIterations; n++) {
      iterate(n, numFeatures, lambda);
    }

    Job uAsMatrix = prepareJob(pathToU(numIterations - 1), new Path(getOutputPath(), "U"),
        SequenceFileInputFormat.class, ToMatrixMapper.class, IntWritable.class, VectorWritable.class, Reducer.class,
        IntWritable.class, VectorWritable.class, SequenceFileOutputFormat.class);
    uAsMatrix.waitForCompletion(true);

    Job mAsMatrix = prepareJob(pathToM(numIterations - 1), new Path(getOutputPath(), "M"),
        SequenceFileInputFormat.class, ToMatrixMapper.class, IntWritable.class, VectorWritable.class, Reducer.class,
        IntWritable.class, VectorWritable.class, SequenceFileOutputFormat.class);
    mAsMatrix.waitForCompletion(true);

    return 0;
  }

  static class ToMatrixMapper
      extends Mapper<VarIntWritable,FeatureVectorWithRatingWritable,IntWritable,VectorWritable> {
    @Override
    protected void map(VarIntWritable key, FeatureVectorWithRatingWritable value, Context ctx) 
      throws IOException, InterruptedException {
      ctx.write(new IntWritable(key.get()), new VectorWritable(value.getFeatureVector()));
    }
  }


  private void iterate(int currentIteration, int numFeatures, double lambda)
      throws IOException, ClassNotFoundException, InterruptedException {
    /* fix M, compute U */
    joinAndSolve(pathToM(currentIteration - 1), pathToItemRatings(), pathToU(currentIteration), numFeatures,
        lambda, currentIteration, STEP_ONE);
    /* fix U, compute M */
    joinAndSolve(pathToU(currentIteration), pathToUserRatings(), pathToM(currentIteration), numFeatures,
        lambda, currentIteration, STEP_TWO);
  }

  private void joinAndSolve(Path featureMatrix, Path ratingMatrix, Path outputPath, int numFeatures, double lambda,
      int currentIteration, String step) throws IOException, ClassNotFoundException, InterruptedException  {

    Path joinPath = new Path(ratingMatrix.toString() + ',' + featureMatrix);
    Path featureVectorWithRatingPath = joinAndSolvePath(currentIteration, step);

    Job joinToFeatureVectorWithRating = prepareJob(joinPath, featureVectorWithRatingPath, SequenceFileInputFormat.class,
        Mapper.class, VarIntWritable.class, FeatureVectorWithRatingWritable.class,
        JoinFeatureVectorAndRatingsReducer.class, IndexedVarIntWritable.class, FeatureVectorWithRatingWritable.class,
        SequenceFileOutputFormat.class);
    joinToFeatureVectorWithRating.waitForCompletion(true);

    Job solve = prepareJob(featureVectorWithRatingPath, outputPath, SequenceFileInputFormat.class, Mapper.class,
        IndexedVarIntWritable.class, FeatureVectorWithRatingWritable.class, SolvingReducer.class, VarIntWritable.class,
        FeatureVectorWithRatingWritable.class, SequenceFileOutputFormat.class);
    Configuration solveConf = solve.getConfiguration();
    solve.setGroupingComparatorClass(IndexedVarIntWritable.GroupingComparator.class);
    solveConf.setInt(NUM_FEATURES, numFeatures);
    solveConf.set(LAMBDA, String.valueOf(lambda));
    solve.waitForCompletion(true);    
  }

  static class PrefsToRatingsMapper
      extends Mapper<LongWritable,Text,VarIntWritable,FeatureVectorWithRatingWritable> {

    private boolean transpose;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      transpose = ctx.getConfiguration().getBoolean(MAP_TRANSPOSED, false);
    }

    @Override
    protected void map(LongWritable offset, Text line, Context ctx) throws IOException, InterruptedException {
      String[] tokens = TasteHadoopUtils.splitPrefTokens(line.toString());
      int keyIDIndex = TasteHadoopUtils.idToIndex(Long.parseLong(tokens[transpose ? 0 : 1]));
      int valueIDIndex = TasteHadoopUtils.idToIndex(Long.parseLong(tokens[transpose ? 1 : 0]));
      float rating = Float.parseFloat(tokens[2]);
      ctx.write(new VarIntWritable(keyIDIndex), new FeatureVectorWithRatingWritable(valueIDIndex, rating));
    }
  }

  static class JoinFeatureVectorAndRatingsReducer
      extends Reducer<VarIntWritable,FeatureVectorWithRatingWritable,IndexedVarIntWritable,FeatureVectorWithRatingWritable> {

    @Override
    protected void reduce(VarIntWritable id, Iterable<FeatureVectorWithRatingWritable> values, Context ctx)
      throws IOException, InterruptedException {
      Vector featureVector = null;
      Map<Integer,Float> ratings = Maps.newHashMap();
      for (FeatureVectorWithRatingWritable value : values) {
        if (value.getFeatureVector() == null) {
          ratings.put(value.getIDIndex(), value.getRating());
        } else {
          featureVector = value.getFeatureVector().clone();          
        }
      }

      if (featureVector == null || ratings.isEmpty()) {
        throw new IllegalStateException("Unable to join data for " + id);
      }      
      for (Map.Entry<Integer,Float> rating : ratings.entrySet()) {
        ctx.write(new IndexedVarIntWritable(rating.getKey(), id.get()),
            new FeatureVectorWithRatingWritable(id.get(), rating.getValue(), featureVector));
      }
    }
  }

  static class SolvingReducer
      extends Reducer<IndexedVarIntWritable,FeatureVectorWithRatingWritable,VarIntWritable,FeatureVectorWithRatingWritable> {

    private int numFeatures;
    private double lambda;
    private AlternateLeastSquaresSolver solver;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      super.setup(ctx);
      lambda = Double.parseDouble(ctx.getConfiguration().get(LAMBDA));
      numFeatures = ctx.getConfiguration().getInt(NUM_FEATURES, -1);
      solver = new AlternateLeastSquaresSolver();

      Preconditions.checkArgument(numFeatures > 0, "numFeatures was not set correctly!");
    }

    @Override
    protected void reduce(IndexedVarIntWritable key, Iterable<FeatureVectorWithRatingWritable> values, Context ctx)
      throws IOException, InterruptedException {
      List<Vector> UorMColumns = Lists.newArrayList();
      Vector ratingVector = new RandomAccessSparseVector(Integer.MAX_VALUE);
      int n = 0;
      for (FeatureVectorWithRatingWritable value : values) {
        ratingVector.setQuick(n++, value.getRating());
        UorMColumns.add(value.getFeatureVector());
      }
      Vector uiOrmj = solver.solve(UorMColumns, new SequentialAccessSparseVector(ratingVector), lambda, numFeatures);
      ctx.write(new VarIntWritable(key.getValue()), new FeatureVectorWithRatingWritable(key.getValue(), uiOrmj));
    }
  }

  static class ItemIDRatingMapper extends Mapper<LongWritable,Text,VarLongWritable,FloatWritable> {
    @Override
    protected void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
      String[] tokens = TasteHadoopUtils.splitPrefTokens(value.toString());
      ctx.write(new VarLongWritable(Long.parseLong(tokens[1])), new FloatWritable(Float.parseFloat(tokens[2])));
    }
  }

  static class InitializeMReducer
      extends Reducer<VarLongWritable,FloatWritable,VarIntWritable,FeatureVectorWithRatingWritable> {

    private int numFeatures;
    private final Random random = RandomUtils.getRandom();

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      super.setup(ctx);
      numFeatures = ctx.getConfiguration().getInt(NUM_FEATURES, -1);

      Preconditions.checkArgument(numFeatures > 0, "numFeatures was not set correctly!");
    }

    @Override
    protected void reduce(VarLongWritable itemID, Iterable<FloatWritable> ratings, Context ctx) 
        throws IOException, InterruptedException {

      RunningAverage averageRating = new FullRunningAverage();
      for (FloatWritable rating : ratings) {
        averageRating.addDatum(rating.get());
      }

      int itemIDIndex = TasteHadoopUtils.idToIndex(itemID.get());
      Vector columnOfM = new DenseVector(numFeatures);

      columnOfM.setQuick(0, averageRating.getAverage());
      for (int n = 1; n < numFeatures; n++) {
        columnOfM.setQuick(n, random.nextDouble());
      }

      ctx.write(new VarIntWritable(itemIDIndex), new FeatureVectorWithRatingWritable(itemIDIndex, columnOfM));
    }
  }

  private Path joinAndSolvePath(int currentIteration, String step) {
    return new Path(tempDir, "joinAndSolve-" + currentIteration + '-' + step);
  }

  private Path pathToM(int iteration) {
    return new Path(tempDir, "M-" + iteration);
  }

  private Path pathToU(int iteration) {
    return new Path(tempDir, "U-" + iteration);
  }

  private Path pathToItemRatings() {
    return new Path(tempDir, "itemsAsFeatureWithRatingWritable");
  }

  private Path pathToUserRatings() {
    return new Path(tempDir, "usersAsFeatureWithRatingWritable");
  }
}
