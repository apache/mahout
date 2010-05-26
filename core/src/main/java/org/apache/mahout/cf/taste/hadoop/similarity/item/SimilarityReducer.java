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

package org.apache.mahout.cf.taste.hadoop.similarity.item;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.CoRating;
import org.apache.mahout.cf.taste.hadoop.similarity.DistributedItemSimilarity;

/**
 * Finally compute the similarity for each item-pair, that has been corated at least once.
 * Computation is done with an external implementation of {@link DistributedItemSimilarity}.
 */
public final class SimilarityReducer extends
    Reducer<ItemPairWritable,CoRating,EntityEntityWritable,DoubleWritable> {

  private DistributedItemSimilarity distributedItemSimilarity;
  private int numberOfUsers;

  @Override
  public void setup(Context context) {
    Configuration jobConf = context.getConfiguration();
    distributedItemSimilarity =
      ItemSimilarityJob.instantiateSimilarity(jobConf.get(ItemSimilarityJob.DISTRIBUTED_SIMILARITY_CLASSNAME));
    numberOfUsers = jobConf.getInt(ItemSimilarityJob.NUMBER_OF_USERS, -1);
    if (numberOfUsers <= 0) {
      throw new IllegalStateException("Number of users was not set correctly");
    }
  }

  @Override
  public void reduce(ItemPairWritable pair,
                     Iterable<CoRating> coRatings,
                     Context context)
      throws IOException, InterruptedException {

    double similarity =
      distributedItemSimilarity.similarity(coRatings, pair.getItemAWeight(), pair.getItemBWeight(), numberOfUsers);

    if (!Double.isNaN(similarity)) {
      context.write(pair.getItemItemWritable(), new DoubleWritable(similarity));
    }
  }

}
