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

package org.apache.mahout.cf.taste.hadoop.item;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.cf.taste.hadoop.MapFilesMap;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.impl.recommender.ByValueRecommendedItemComparator;
import org.apache.mahout.cf.taste.impl.recommender.GenericRecommendedItem;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.math.RandomAccessSparseVectorWritable;
import org.apache.mahout.math.Vector;

public final class AggregateAndRecommendReducer extends MapReduceBase implements
    Reducer<LongWritable,RandomAccessSparseVectorWritable,LongWritable,RecommendedItemsWritable> {

  static final String ITEMID_INDEX_PATH = "itemIDIndexPath";
  static final String RECOMMENDATIONS_PER_USER = "recommendationsPerUser";

  private int recommendationsPerUser;
  private MapFilesMap<IntWritable,LongWritable> indexItemIDMap;

  @Override
  public void configure(JobConf jobConf) {
    try {
      FileSystem fs = FileSystem.get(jobConf);
      Path itemIDIndexPath = new Path(jobConf.get(ITEMID_INDEX_PATH)).makeQualified(fs);
      recommendationsPerUser = jobConf.getInt(RECOMMENDATIONS_PER_USER, 10);
      indexItemIDMap = new MapFilesMap<IntWritable,LongWritable>(fs, itemIDIndexPath, new Configuration());
    } catch (IOException ioe) {
      throw new IllegalStateException(ioe);
    }
  }

  @Override
  public void reduce(LongWritable key,
                     Iterator<RandomAccessSparseVectorWritable> values,
                     OutputCollector<LongWritable, RecommendedItemsWritable> output,
                     Reporter reporter) throws IOException {
    if (!values.hasNext()) {
      return;
    }
    Vector recommendationVector = values.next().get();
    while (values.hasNext()) {
      recommendationVector = recommendationVector.plus(values.next().get());
    }

    Queue<RecommendedItem> topItems = new PriorityQueue<RecommendedItem>(recommendationsPerUser + 1,
    Collections.reverseOrder(ByValueRecommendedItemComparator.getInstance()));

    Iterator<Vector.Element> recommendationVectorIterator = recommendationVector.iterateNonZero();
    LongWritable itemID = new LongWritable();
    while (recommendationVectorIterator.hasNext()) {
      Vector.Element element = recommendationVectorIterator.next();
      int index = element.index();
      if (topItems.size() < recommendationsPerUser) {
        LongWritable theItemID = indexItemIDMap.get(new IntWritable(index), itemID);
        if (theItemID != null) {
          topItems.add(new GenericRecommendedItem(theItemID.get(), (float) element.get()));
        } // else, huh?
      } else if (element.get() > topItems.peek().getValue()) {
        LongWritable theItemID = indexItemIDMap.get(new IntWritable(index), itemID);
        if (theItemID != null) {
          topItems.add(new GenericRecommendedItem(theItemID.get(), (float) element.get()));
          topItems.poll();
        } // else, huh?
      }
    }

    if (!topItems.isEmpty()) {
      List<RecommendedItem> recommendations = new ArrayList<RecommendedItem>(topItems.size());
      recommendations.addAll(topItems);
      Collections.sort(recommendations, ByValueRecommendedItemComparator.getInstance());
      output.collect(key, new RecommendedItemsWritable(recommendations));
    }
  }

}