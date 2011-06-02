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

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.Map;

/**
 * <p>Compute predictions for user,item pairs using an existing matrix factorization</p>
 *
 * <p>Command line arguments specific to this class are:</p>
 *
 * <ol>
 * <li>--output (path): path where output should go</li>
 * <li>--pairs (path): path containing the test ratings, each line must be userID,itemID</li>
 * <li>--userFeatures (path): path to the user feature matrix</li>
 * <li>--itemFeatures (path): path to the item feature matrix</li>
 * </ol>
 */
public class PredictionJob extends AbstractJob {

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new PredictionJob(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addOption("pairs", "p", "path containing the test ratings, each line must be: userID,itemID", true);
    addOption("userFeatures", "u", "path to the user feature matrix", true);
    addOption("itemFeatures", "i", "path to the item feature matrix", true);
    addOutputOption();

    Map<String,String> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    Path pairs = new Path(parsedArgs.get("--pairs"));
    Path userFeatures = new Path(parsedArgs.get("--userFeatures"));
    Path itemFeatures = new Path(parsedArgs.get("--itemFeatures"));

    Path tempDirPath = new Path(parsedArgs.get("--tempDir"));

    Path convertedPairs = new Path(tempDirPath, "convertedPairs");
    Path convertedUserFeatures = new Path(tempDirPath, "convertedUserFeatures");
    Path convertedItemFeatures = new Path(tempDirPath, "convertedItemFeatures");

    Path pairsJoinedWithItemFeatures = new Path(tempDirPath, "pairsJoinedWithItemFeatures");

    /* joins here could spare more than 50% of their M/R cycles when MultipleInputs is available again */
    Job convertPairs = prepareJob(pairs, convertedPairs, TextInputFormat.class, PairsMapper.class,
        TaggedVarIntWritable.class, VectorWithIndexWritable.class, Reducer.class, TaggedVarIntWritable.class,
        VectorWithIndexWritable.class, SequenceFileOutputFormat.class);
    convertPairs.waitForCompletion(true);

    Job convertUserFeatures = prepareJob(userFeatures, convertedUserFeatures, SequenceFileInputFormat.class,
        FeaturesMapper.class, TaggedVarIntWritable.class, VectorWithIndexWritable.class, Reducer.class,
        TaggedVarIntWritable.class, VectorWithIndexWritable.class, SequenceFileOutputFormat.class);
    convertUserFeatures.waitForCompletion(true);

    Job convertItemFeatures = prepareJob(itemFeatures, convertedItemFeatures, SequenceFileInputFormat.class,
        FeaturesMapper.class, TaggedVarIntWritable.class, VectorWithIndexWritable.class, Reducer.class,
        TaggedVarIntWritable.class, VectorWithIndexWritable.class, SequenceFileOutputFormat.class);
    convertItemFeatures.waitForCompletion(true);

    Job joinPairsWithItemFeatures = prepareJob(new Path(convertedPairs + "," + convertedItemFeatures),
        pairsJoinedWithItemFeatures, SequenceFileInputFormat.class, Mapper.class, TaggedVarIntWritable.class,
        VectorWithIndexWritable.class, JoinProbesWithItemFeaturesReducer.class, TaggedVarIntWritable.class,
        VectorWithIndexWritable.class, SequenceFileOutputFormat.class);
    joinPairsWithItemFeatures.setPartitionerClass(HashPartitioner.class);
    joinPairsWithItemFeatures.setGroupingComparatorClass(TaggedVarIntWritable.GroupingComparator.class);
    joinPairsWithItemFeatures.waitForCompletion(true);

    Job predictRatings = prepareJob(new Path(pairsJoinedWithItemFeatures + "," + convertedUserFeatures),
        getOutputPath(), SequenceFileInputFormat.class, Mapper.class, TaggedVarIntWritable.class,
        VectorWithIndexWritable.class, PredictRatingReducer.class, Text.class, NullWritable.class,
        TextOutputFormat.class);
    predictRatings.setPartitionerClass(HashPartitioner.class);
    predictRatings.setGroupingComparatorClass(TaggedVarIntWritable.GroupingComparator.class);
    predictRatings.waitForCompletion(true);

    return 0;
  }

  public static class PairsMapper extends Mapper<LongWritable,Text,TaggedVarIntWritable,VectorWithIndexWritable> {
    @Override
    protected void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
      String[] tokens = TasteHadoopUtils.splitPrefTokens(value.toString());
      int userIDIndex = TasteHadoopUtils.idToIndex(Long.parseLong(tokens[0]));
      int itemIDIndex = TasteHadoopUtils.idToIndex(Long.parseLong(tokens[1]));
      ctx.write(new TaggedVarIntWritable(itemIDIndex, false), new VectorWithIndexWritable(userIDIndex));
    }
  }

  public static class FeaturesMapper
      extends Mapper<IntWritable,VectorWritable,TaggedVarIntWritable,VectorWithIndexWritable> {
    @Override
    protected void map(IntWritable id, VectorWritable features, Context ctx) throws IOException, InterruptedException {
      ctx.write(new TaggedVarIntWritable(id.get(), true), new VectorWithIndexWritable(features.get()));
    }
  }

  public static class JoinProbesWithItemFeaturesReducer
      extends Reducer<TaggedVarIntWritable,VectorWithIndexWritable,TaggedVarIntWritable,VectorWithIndexWritable> {
    @Override
    protected void reduce(TaggedVarIntWritable key, Iterable<VectorWithIndexWritable> values, Context ctx)
      throws IOException, InterruptedException {
      int itemIDIndex = key.get();
      Vector itemFeatures = null;
      for (VectorWithIndexWritable vectorWithIndexWritable : values) {
        if (itemFeatures == null && vectorWithIndexWritable.getVector() != null) {
          itemFeatures = vectorWithIndexWritable.getVector();
        } else if (itemFeatures == null && vectorWithIndexWritable.getVector() == null) {
          /* no feature vector is found for that item */
          return;
        } else {
          int userIDIndex = vectorWithIndexWritable.getIDIndex();
          ctx.write(new TaggedVarIntWritable(userIDIndex, false),
              new VectorWithIndexWritable(itemIDIndex, itemFeatures));
        }
      }
    }
  }

  public static class PredictRatingReducer
      extends Reducer<TaggedVarIntWritable,VectorWithIndexWritable,Text,NullWritable> {
    @Override
    protected void reduce(TaggedVarIntWritable key, Iterable<VectorWithIndexWritable> values, Context ctx)
      throws IOException, InterruptedException {
      Vector userFeatures = null;
      int userIDIndex = key.get();
      for (VectorWithIndexWritable vectorWithIndexWritable : values) {
        if (userFeatures == null && vectorWithIndexWritable.getVector() != null) {
          userFeatures = vectorWithIndexWritable.getVector();
        } else if (userFeatures == null && vectorWithIndexWritable.getVector() == null) {
          /* no feature vector is found for that user */
          return;
        } else {
          int itemIDIndex = vectorWithIndexWritable.getIDIndex();
          Vector itemFeatures = vectorWithIndexWritable.getVector();
          double estimatedPrediction = userFeatures.dot(itemFeatures);
          ctx.write(new Text(userIDIndex + "," + itemIDIndex + ',' + estimatedPrediction), NullWritable.get());
        }
      }
    }
  }

}
