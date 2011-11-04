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

import com.google.common.collect.Lists;
import com.google.common.primitives.Floats;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.common.TopK;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.impl.recommender.GenericRecommendedItem;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.IntObjectProcedure;
import org.apache.mahout.math.map.OpenIntObjectHashMap;
import org.apache.mahout.math.set.OpenIntHashSet;

import java.io.IOException;
import java.util.Comparator;
import java.util.Iterator;
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
 * <li>--numRecommendations (int): maximum number of recommendations per user</li>
 * <li>--maxRating (double): maximum rating of an item</li>
 * <li>--numFeatures (int): number of features to use for decomposition </li>
 * </ol>
 */
public class RecommenderJob extends AbstractJob {

  private static final String NUM_RECOMMENDATIONS = RecommenderJob.class.getName() + ".numRecommendations";
  private static final String USER_FEATURES_PATH = RecommenderJob.class.getName() + ".userFeatures";
  private static final String ITEM_FEATURES_PATH = RecommenderJob.class.getName() + ".itemFeatures";
  private static final String MAX_RATING = RecommenderJob.class.getName() + ".maxRating";

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
    addOutputOption();

    Map<String,String> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    Job prediction = prepareJob(getInputPath(), getOutputPath(), SequenceFileInputFormat.class, PredictionMapper.class,
        IntWritable.class, RecommendedItemsWritable.class, TextOutputFormat.class);
    prediction.getConfiguration().setInt(NUM_RECOMMENDATIONS,
        Integer.parseInt(parsedArgs.get("--numRecommendations")));
    prediction.getConfiguration().set(USER_FEATURES_PATH, parsedArgs.get("--userFeatures"));
    prediction.getConfiguration().set(ITEM_FEATURES_PATH, parsedArgs.get("--itemFeatures"));
    prediction.getConfiguration().set(MAX_RATING, parsedArgs.get("--maxRating"));
    prediction.waitForCompletion(true);

    return 0;
  }

  private static final Comparator<RecommendedItem> BY_PREFERENCE_VALUE =
      new Comparator<RecommendedItem>() {
        @Override
        public int compare(RecommendedItem one, RecommendedItem two) {
          return Floats.compare(one.getValue(), two.getValue());
        }
      };

  static class PredictionMapper
      extends Mapper<IntWritable,VectorWritable,IntWritable,RecommendedItemsWritable> {

    private OpenIntObjectHashMap<Vector> U;
    private OpenIntObjectHashMap<Vector> M;

    private int recommendationsPerUser;
    private float maxRating;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      recommendationsPerUser = ctx.getConfiguration().getInt(NUM_RECOMMENDATIONS,
          DEFAULT_NUM_RECOMMENDATIONS);

      Path pathToU = new Path(ctx.getConfiguration().get(USER_FEATURES_PATH));
      Path pathToM = new Path(ctx.getConfiguration().get(ITEM_FEATURES_PATH));

      U = ALSUtils.readMatrixByRows(pathToU, ctx.getConfiguration());
      M = ALSUtils.readMatrixByRows(pathToM, ctx.getConfiguration());

      maxRating = Float.parseFloat(ctx.getConfiguration().get(MAX_RATING));
    }

    @Override
    protected void map(IntWritable userIDWritable, VectorWritable ratingsWritable, Context ctx)
        throws IOException, InterruptedException {

      Vector ratings = ratingsWritable.get();
      final int userID = userIDWritable.get();
      final OpenIntHashSet alreadyRatedItems = new OpenIntHashSet(ratings.getNumNondefaultElements());
      final TopK<RecommendedItem> topKItems = new TopK<RecommendedItem>(recommendationsPerUser, BY_PREFERENCE_VALUE);

      Iterator<Vector.Element> ratingsIterator = ratings.iterateNonZero();
      while (ratingsIterator.hasNext()) {
        alreadyRatedItems.add(ratingsIterator.next().index());
      }

      M.forEachPair(new IntObjectProcedure<Vector>() {
        @Override
        public boolean apply(int itemID, Vector itemFeatures) {
          if (!alreadyRatedItems.contains(itemID)) {
            double predictedRating = U.get(userID).dot(itemFeatures);
            topKItems.offer(new GenericRecommendedItem(itemID, (float) predictedRating));
          }
          return true;
        }
      });

      List<RecommendedItem> recommendedItems = Lists.newArrayListWithExpectedSize(recommendationsPerUser);
      for (RecommendedItem topItem : topKItems.retrieve()) {
        recommendedItems.add(new GenericRecommendedItem(topItem.getItemID(), Math.min(topItem.getValue(), maxRating)));
      }

      if (!topKItems.isEmpty()) {
        ctx.write(userIDWritable, new RecommendedItemsWritable(recommendedItems));
      }
    }
  }
}
