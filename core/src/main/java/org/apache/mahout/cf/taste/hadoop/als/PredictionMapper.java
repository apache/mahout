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
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.mahout.cf.taste.hadoop.MutableRecommendedItem;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.hadoop.TopItemsQueue;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.IntObjectProcedure;
import org.apache.mahout.math.map.OpenIntLongHashMap;
import org.apache.mahout.math.map.OpenIntObjectHashMap;
import org.apache.mahout.math.set.OpenIntHashSet;

import java.io.IOException;
import java.util.List;

/**
 * a multithreaded mapper that loads the feature matrices U and M into memory. Afterwards it computes recommendations
 * from these. Can be executed by a {@link MultithreadedSharingMapper}.
 */
public class PredictionMapper extends SharingMapper<IntWritable,VectorWritable,LongWritable,RecommendedItemsWritable,
    Pair<OpenIntObjectHashMap<Vector>,OpenIntObjectHashMap<Vector>>> {

  private int recommendationsPerUser;
  private float maxRating;

  private boolean usesLongIDs;
  private OpenIntLongHashMap userIDIndex;
  private OpenIntLongHashMap itemIDIndex;

  private final LongWritable userIDWritable = new LongWritable();
  private final RecommendedItemsWritable recommendations = new RecommendedItemsWritable();

  @Override
  Pair<OpenIntObjectHashMap<Vector>, OpenIntObjectHashMap<Vector>> createSharedInstance(Context ctx) {
    Configuration conf = ctx.getConfiguration();
    Path pathToU = new Path(conf.get(RecommenderJob.USER_FEATURES_PATH));
    Path pathToM = new Path(conf.get(RecommenderJob.ITEM_FEATURES_PATH));

    OpenIntObjectHashMap<Vector> U = ALS.readMatrixByRows(pathToU, conf);
    OpenIntObjectHashMap<Vector> M = ALS.readMatrixByRows(pathToM, conf);

    return new Pair<OpenIntObjectHashMap<Vector>, OpenIntObjectHashMap<Vector>>(U, M);
  }

  @Override
  protected void setup(Context ctx) throws IOException, InterruptedException {
    Configuration conf = ctx.getConfiguration();
    recommendationsPerUser = conf.getInt(RecommenderJob.NUM_RECOMMENDATIONS,
        RecommenderJob.DEFAULT_NUM_RECOMMENDATIONS);
    maxRating = Float.parseFloat(conf.get(RecommenderJob.MAX_RATING));

    usesLongIDs = conf.getBoolean(ParallelALSFactorizationJob.USES_LONG_IDS, false);
    if (usesLongIDs) {
      userIDIndex = TasteHadoopUtils.readIDIndexMap(conf.get(RecommenderJob.USER_INDEX_PATH), conf);
      itemIDIndex = TasteHadoopUtils.readIDIndexMap(conf.get(RecommenderJob.ITEM_INDEX_PATH), conf);
    }
  }

  @Override
  protected void map(IntWritable userIndexWritable, VectorWritable ratingsWritable, Context ctx)
    throws IOException, InterruptedException {

    Pair<OpenIntObjectHashMap<Vector>, OpenIntObjectHashMap<Vector>> uAndM = getSharedInstance();
    OpenIntObjectHashMap<Vector> U = uAndM.getFirst();
    OpenIntObjectHashMap<Vector> M = uAndM.getSecond();

    Vector ratings = ratingsWritable.get();
    int userIndex = userIndexWritable.get();
    final OpenIntHashSet alreadyRatedItems = new OpenIntHashSet(ratings.getNumNondefaultElements());

    for (Vector.Element e : ratings.nonZeroes()) {
      alreadyRatedItems.add(e.index());
    }

    final TopItemsQueue topItemsQueue = new TopItemsQueue(recommendationsPerUser);
    final Vector userFeatures = U.get(userIndex);

    M.forEachPair(new IntObjectProcedure<Vector>() {
      @Override
      public boolean apply(int itemID, Vector itemFeatures) {
        if (!alreadyRatedItems.contains(itemID)) {
          double predictedRating = userFeatures.dot(itemFeatures);

          MutableRecommendedItem top = topItemsQueue.top();
          if (predictedRating > top.getValue()) {
            top.set(itemID, (float) predictedRating);
            topItemsQueue.updateTop();
          }
        }
        return true;
      }
    });

    List<RecommendedItem> recommendedItems = topItemsQueue.getTopItems();

    if (!recommendedItems.isEmpty()) {

      // cap predictions to maxRating
      for (RecommendedItem topItem : recommendedItems) {
        ((MutableRecommendedItem) topItem).capToMaxValue(maxRating);
      }

      if (usesLongIDs) {
        long userID = userIDIndex.get(userIndex);
        userIDWritable.set(userID);

        for (RecommendedItem topItem : recommendedItems) {
          // remap item IDs
          long itemID = itemIDIndex.get((int) topItem.getItemID());
          ((MutableRecommendedItem) topItem).setItemID(itemID);
        }

      } else {
        userIDWritable.set(userIndex);
      }

      recommendations.set(recommendedItems);
      ctx.write(userIDWritable, recommendations);
    }
  }
}
