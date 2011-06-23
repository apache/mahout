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

package org.apache.mahout.cf.taste.hadoop.als.eval;

import com.google.common.io.Closeables;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.hadoop.als.PredictionJob;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.IntPairWritable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Map;

/**
 * <p>Measures the root-mean-squared error of a ratring matrix factorization against a test set.</p>
 *
 * <p>Command line arguments specific to this class are:</p>
 *
 * <ol>
 * <li>--output (path): path where output should go</li>
 * <li>--pairs (path): path containing the test ratings, each line must be userID,itemID,rating</li>
 * <li>--userFeatures (path): path to the user feature matrix</li>
 * <li>--itemFeatures (path): path to the item feature matrix</li>
 * </ol>
 */
public class ParallelFactorizationEvaluator extends AbstractJob {

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new ParallelFactorizationEvaluator(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addOption("pairs", "p", "path containing the test ratings, each line must be userID,itemID,rating", true);
    addOption("userFeatures", "u", "path to the user feature matrix", true);
    addOption("itemFeatures", "i", "path to the item feature matrix", true);
    addOutputOption();

    Map<String,String> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    Path tempDir = new Path(parsedArgs.get("--tempDir"));
    Path predictions = new Path(tempDir, "predictions");
    Path errors = new Path(tempDir, "errors");

    ToolRunner.run(getConf(), new PredictionJob(), new String[] { "--output", predictions.toString(),
        "--pairs", parsedArgs.get("--pairs"), "--userFeatures", parsedArgs.get("--userFeatures"),
        "--itemFeatures", parsedArgs.get("--itemFeatures"),
        "--tempDir", tempDir.toString() });

    Job estimationErrors = prepareJob(new Path(parsedArgs.get("--pairs") + ',' + predictions), errors,
        TextInputFormat.class, PairsWithRatingMapper.class, IntPairWritable.class, DoubleWritable.class,
        ErrorReducer.class, DoubleWritable.class, NullWritable.class, SequenceFileOutputFormat.class);
    estimationErrors.waitForCompletion(true);

    BufferedWriter writer  = null;
    try {
      FileSystem fs = FileSystem.get(getOutputPath().toUri(), getConf());
      FSDataOutputStream outputStream = fs.create(new Path(getOutputPath(), "rmse.txt"));
      double rmse = computeRmse(errors);
      writer = new BufferedWriter(new OutputStreamWriter(outputStream));
      writer.write(String.valueOf(rmse));
    } finally {
      Closeables.closeQuietly(writer);
    }

    return 0;
  }

  protected double computeRmse(Path errors) {
    RunningAverage average = new FullRunningAverage();
    for (Pair<DoubleWritable,NullWritable> entry :
        new SequenceFileDirIterable<DoubleWritable, NullWritable>(errors,
                                                                  PathType.LIST,
                                                                  PathFilters.logsCRCFilter(),
                                                                  getConf())) {
      DoubleWritable error = entry.getFirst();
      average.addDatum(error.get() * error.get());
    }

    return Math.sqrt(average.getAverage());
  }

  public static class PairsWithRatingMapper extends Mapper<LongWritable,Text,IntPairWritable,DoubleWritable> {
    @Override
    protected void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
      String[] tokens = TasteHadoopUtils.splitPrefTokens(value.toString());
      int userIDIndex = TasteHadoopUtils.idToIndex(Long.parseLong(tokens[0]));
      int itemIDIndex = TasteHadoopUtils.idToIndex(Long.parseLong(tokens[1]));
      double rating = Double.parseDouble(tokens[2]);
      ctx.write(new IntPairWritable(userIDIndex, itemIDIndex), new DoubleWritable(rating));
    }
  }

  public static class ErrorReducer extends Reducer<IntPairWritable,DoubleWritable,DoubleWritable,NullWritable> {
    @Override
    protected void reduce(IntPairWritable key, Iterable<DoubleWritable> ratingAndEstimate, Context ctx)
        throws IOException, InterruptedException {

      double error = Double.NaN;
      boolean bothFound = false;
      for (DoubleWritable ratingOrEstimate : ratingAndEstimate) {
        if (Double.isNaN(error)) {
          error = ratingOrEstimate.get();
        } else {
          error -= ratingOrEstimate.get();
          bothFound = true;
          break;
        }
      }

      if (bothFound) {
        ctx.write(new DoubleWritable(error), NullWritable.get());
      }
    }
  }
}