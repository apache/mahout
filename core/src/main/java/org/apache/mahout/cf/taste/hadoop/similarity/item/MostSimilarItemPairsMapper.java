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

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.cf.taste.common.TopK;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.map.OpenIntLongHashMap;

import java.io.IOException;
import java.util.Iterator;

public final class MostSimilarItemPairsMapper
    extends Mapper<IntWritable,VectorWritable,EntityEntityWritable,DoubleWritable> {

  private OpenIntLongHashMap indexItemIDMap;
  private int maxSimilarItemsPerItem;

  @Override
  protected void setup(Context ctx) {
    Configuration conf = ctx.getConfiguration();
    maxSimilarItemsPerItem = conf.getInt(ItemSimilarityJob.MAX_SIMILARITIES_PER_ITEM, -1);
    indexItemIDMap = TasteHadoopUtils.readItemIDIndexMap(conf.get(ItemSimilarityJob.ITEM_ID_INDEX_PATH_STR), conf);

    Preconditions.checkArgument(maxSimilarItemsPerItem > 0, "maxSimilarItemsPerItem was not correctly set!");
  }

  @Override
  protected void map(IntWritable itemIDIndexWritable, VectorWritable similarityVector, Context ctx)
    throws IOException, InterruptedException {

    int itemIDIndex = itemIDIndexWritable.get();

    TopK<SimilarItem> topKMostSimilarItems =
        new TopK<SimilarItem>(maxSimilarItemsPerItem, SimilarItem.COMPARE_BY_SIMILARITY);

    Iterator<Vector.Element> similarityVectorIterator = similarityVector.get().iterateNonZero();

    while (similarityVectorIterator.hasNext()) {
      Vector.Element element = similarityVectorIterator.next();
      /* ignore self similarities */
      if (element.index() != itemIDIndex) {
        topKMostSimilarItems.offer(new SimilarItem(indexItemIDMap.get(element.index()), element.get()));
      }
    }

    long itemID = indexItemIDMap.get(itemIDIndex);
    for (SimilarItem similarItem : topKMostSimilarItems.retrieve()) {
      long otherItemID = similarItem.getItemID();
      if (itemID < otherItemID) {
        ctx.write(new EntityEntityWritable(itemID, otherItemID), new DoubleWritable(similarItem.getSimilarity()));
      } else {
        ctx.write(new EntityEntityWritable(otherItemID, itemID), new DoubleWritable(similarItem.getSimilarity()));
      }
    }

  }
}
