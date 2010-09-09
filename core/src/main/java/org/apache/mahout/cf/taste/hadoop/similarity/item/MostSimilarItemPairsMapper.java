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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.map.OpenIntLongHashMap;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;

public final class MostSimilarItemPairsMapper
    extends Mapper<IntWritable,VectorWritable,EntityEntityWritable,DoubleWritable> {

  private OpenIntLongHashMap indexItemIDMap;
  private int maxSimilarItemsPerItem;

  @Override
  protected void setup(Context ctx) {
    Configuration conf = ctx.getConfiguration();
    //String itemIDIndexPathStr = conf.get(ItemSimilarityJob.ITEM_ID_INDEX_PATH_STR);
    maxSimilarItemsPerItem = conf.getInt(ItemSimilarityJob.MAX_SIMILARITIES_PER_ITEM, -1);
    if (maxSimilarItemsPerItem < 1) {
      throw new IllegalStateException("maxSimilarItemsPerItem was not correctly set!");
    }
    indexItemIDMap = TasteHadoopUtils.readItemIDIndexMap(conf.get(ItemSimilarityJob.ITEM_ID_INDEX_PATH_STR), conf);
  }

  @Override
  protected void map(IntWritable itemIDIndexWritable, VectorWritable similarityVector, Context ctx)
    throws IOException, InterruptedException {

    int itemIDIndex = itemIDIndexWritable.get();

    Queue<SimilarItem> topMostSimilarItems = new PriorityQueue<SimilarItem>(maxSimilarItemsPerItem + 1,
        Collections.reverseOrder(SimilarItem.COMPARE_BY_SIMILARITY));

    Iterator<Vector.Element> similarityVectorIterator = similarityVector.get().iterateNonZero();

    while (similarityVectorIterator.hasNext()) {
      Vector.Element element = similarityVectorIterator.next();
      int index = element.index();
      double value = element.get();
      /* ignore self similarities */
      if (index != itemIDIndex) {
        if (topMostSimilarItems.size() < maxSimilarItemsPerItem) {
          topMostSimilarItems.add(new SimilarItem(indexItemIDMap.get(index), value));
        } else if (value > topMostSimilarItems.peek().getSimilarity()) {
          topMostSimilarItems.add(new SimilarItem(indexItemIDMap.get(index), value));
          topMostSimilarItems.poll();
        }
      }
    }

    if (!topMostSimilarItems.isEmpty()) {
      List<SimilarItem> mostSimilarItems = new ArrayList<SimilarItem>(topMostSimilarItems.size());
      mostSimilarItems.addAll(topMostSimilarItems);
      Collections.sort(mostSimilarItems, SimilarItem.COMPARE_BY_SIMILARITY);

      long itemID = indexItemIDMap.get(itemIDIndex);
      for (SimilarItem similarItem : mostSimilarItems) {
        long otherItemID = similarItem.getItemID();
        if (itemID < otherItemID) {
          ctx.write(new EntityEntityWritable(itemID, otherItemID), new DoubleWritable(similarItem.getSimilarity()));
        } else {
          ctx.write(new EntityEntityWritable(otherItemID, itemID), new DoubleWritable(similarItem.getSimilarity()));
        }
      }
    }
  }
}
