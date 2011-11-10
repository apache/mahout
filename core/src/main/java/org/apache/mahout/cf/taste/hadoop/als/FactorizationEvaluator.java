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
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.map.OpenIntObjectHashMap;

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
public class FactorizationEvaluator extends AbstractJob {

  private static final String USER_FEATURES_PATH = RecommenderJob.class.getName() + ".userFeatures";
  private static final String ITEM_FEATURES_PATH = RecommenderJob.class.getName() + ".itemFeatures";

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new FactorizationEvaluator(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOption("userFeatures", null, "path to the user feature matrix", true);
    addOption("itemFeatures", null, "path to the item feature matrix", true);
    addOutputOption();

    Map<String,String> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    Path errors = getTempPath("errors");

    Job predictRatings = prepareJob(getInputPath(), errors, TextInputFormat.class, PredictRatingsMapper.class,
        DoubleWritable.class, NullWritable.class, SequenceFileOutputFormat.class);
    predictRatings.getConfiguration().set(USER_FEATURES_PATH, parsedArgs.get("--userFeatures"));
    predictRatings.getConfiguration().set(ITEM_FEATURES_PATH, parsedArgs.get("--itemFeatures"));
    predictRatings.waitForCompletion(true);

    BufferedWriter writer  = null;
    try {
      FileSystem fs = FileSystem.get(getOutputPath().toUri(), getConf());
      FSDataOutputStream outputStream = fs.create(getOutputPath("rmse.txt"));
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
      new SequenceFileDirIterable<DoubleWritable, NullWritable>(errors, PathType.LIST, PathFilters.logsCRCFilter(),
          getConf())) {
      DoubleWritable error = entry.getFirst();
      average.addDatum(error.get() * error.get());
    }

    return Math.sqrt(average.getAverage());
  }

  public static class PredictRatingsMapper extends Mapper<LongWritable,Text,DoubleWritable,NullWritable> {

    private OpenIntObjectHashMap<Vector> U;
    private OpenIntObjectHashMap<Vector> M;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      Path pathToU = new Path(ctx.getConfiguration().get(USER_FEATURES_PATH));
      Path pathToM = new Path(ctx.getConfiguration().get(ITEM_FEATURES_PATH));

      U = ALSUtils.readMatrixByRows(pathToU, ctx.getConfiguration());
      M = ALSUtils.readMatrixByRows(pathToM, ctx.getConfiguration());
    }

    @Override
    protected void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {

      String[] tokens = TasteHadoopUtils.splitPrefTokens(value.toString());
      int userID = Integer.parseInt(tokens[0]);
      int itemID = Integer.parseInt(tokens[1]);
      double rating = Double.parseDouble(tokens[2]);

      if (U.containsKey(userID) && M.containsKey(itemID)) {
        double estimate = U.get(userID).dot(M.get(itemID));
        double err = rating - estimate;
        ctx.write(new DoubleWritable(err), NullWritable.get());
      }
    }
  }

}
