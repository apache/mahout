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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.map.MultithreadedMapper;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.common.AbstractJob;

import java.util.List;
import java.util.Map;

/**
 * <p>Computes the top-N recommendations per user from a decomposition of the rating matrix</p>
 *
 * <p>Command line arguments specific to this class are:</p>
 *
 * <ol>
 * <li>--input (path): Directory containing the vectorized user ratings</li>
 * <li>--output (path): path where output should go</li>
 * <li>--numRecommendations (int): maximum number of recommendations per user (default: 10)</li>
 * <li>--maxRating (double): maximum rating of an item</li>
 * <li>--numThreads (int): threads to use per mapper, (default: 1)</li>
 * </ol>
 */
public class RecommenderJob extends AbstractJob {

  static final String NUM_RECOMMENDATIONS = RecommenderJob.class.getName() + ".numRecommendations";
  static final String USER_FEATURES_PATH = RecommenderJob.class.getName() + ".userFeatures";
  static final String ITEM_FEATURES_PATH = RecommenderJob.class.getName() + ".itemFeatures";
  static final String MAX_RATING = RecommenderJob.class.getName() + ".maxRating";
  static final String USER_INDEX_PATH = RecommenderJob.class.getName() + ".userIndex";
  static final String ITEM_INDEX_PATH = RecommenderJob.class.getName() + ".itemIndex";

  static final int DEFAULT_NUM_RECOMMENDATIONS = 10;

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new RecommenderJob(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOption("userFeatures", null, "path to the user feature matrix", true);
    addOption("itemFeatures", null, "path to the item feature matrix", true);
    addOption("numRecommendations", null, "number of recommendations per user",
        String.valueOf(DEFAULT_NUM_RECOMMENDATIONS));
    addOption("maxRating", null, "maximum rating available", true);
    addOption("numThreads", null, "threads per mapper", String.valueOf(1));
    addOption("usesLongIDs", null, "input contains long IDs that need to be translated");
    addOption("userIDIndex", null, "index for user long IDs (necessary if usesLongIDs is true)");
    addOption("itemIDIndex", null, "index for user long IDs (necessary if usesLongIDs is true)");
    addOutputOption();

    Map<String,List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    Job prediction = prepareJob(getInputPath(), getOutputPath(), SequenceFileInputFormat.class,
        MultithreadedSharingMapper.class, IntWritable.class, RecommendedItemsWritable.class, TextOutputFormat.class);
    Configuration conf = prediction.getConfiguration();

    int numThreads = Integer.parseInt(getOption("numThreads"));

    conf.setInt(NUM_RECOMMENDATIONS, Integer.parseInt(getOption("numRecommendations")));
    conf.set(USER_FEATURES_PATH, getOption("userFeatures"));
    conf.set(ITEM_FEATURES_PATH, getOption("itemFeatures"));
    conf.set(MAX_RATING, getOption("maxRating"));

    boolean usesLongIDs = Boolean.parseBoolean(getOption("usesLongIDs"));
    if (usesLongIDs) {
      conf.set(ParallelALSFactorizationJob.USES_LONG_IDS, String.valueOf(true));
      conf.set(USER_INDEX_PATH, getOption("userIDIndex"));
      conf.set(ITEM_INDEX_PATH, getOption("itemIDIndex"));
    }

    MultithreadedMapper.setMapperClass(prediction, PredictionMapper.class);
    MultithreadedMapper.setNumberOfThreads(prediction, numThreads);

    boolean succeeded = prediction.waitForCompletion(true);
    if (!succeeded) {
      return -1;
    }

    return 0;
  }

}
