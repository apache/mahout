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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.cf.taste.hadoop.MapFilesMap;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.impl.recommender.GenericRecommendedItem;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.matrix.SparseVector;
import org.apache.mahout.matrix.Vector;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;

public final class RecommenderMapper
    extends MapReduceBase
    implements Mapper<LongWritable, SparseVector, LongWritable, RecommendedItemsWritable> {

  static final String COOCCURRENCE_PATH = "cooccurrencePath";
  static final String ITEMID_INDEX_PATH = "itemIDIndexPath";
  static final String RECOMMENDATIONS_PER_USER = "recommendationsPerUser";

  private int recommendationsPerUser;
  private MapFilesMap<IntWritable,LongWritable> indexItemIDMap;
  private MapFilesMap<IntWritable,Vector> cooccurrenceColumnMap;

  @Override
  public void configure(JobConf jobConf) {
    try {
      FileSystem fs = FileSystem.get(jobConf);
      Path cooccurrencePath = new Path(jobConf.get(COOCCURRENCE_PATH)).makeQualified(fs);
      Path itemIDIndexPath = new Path(jobConf.get(ITEMID_INDEX_PATH)).makeQualified(fs);
      recommendationsPerUser = jobConf.getInt(RECOMMENDATIONS_PER_USER, 10);
      indexItemIDMap = new MapFilesMap<IntWritable,LongWritable>(fs, itemIDIndexPath, new Configuration());
      cooccurrenceColumnMap = new MapFilesMap<IntWritable,Vector>(fs, cooccurrencePath, new Configuration());
    } catch (IOException ioe) {
      throw new IllegalStateException(ioe);
    }
  }

  @Override
  public void map(LongWritable userID,
                  SparseVector userVector,
                  OutputCollector<LongWritable, RecommendedItemsWritable> output,
                  Reporter reporter) throws IOException {

    Iterator<Vector.Element> userVectorIterator = userVector.iterateNonZero();
    Vector recommendationVector = new SparseVector(Integer.MAX_VALUE, 1000);
    Vector columnVector = new SparseVector(Integer.MAX_VALUE, 1000);
    while (userVectorIterator.hasNext()) {
      Vector.Element element = userVectorIterator.next();
      int index = element.index();
      double value = element.get();
      cooccurrenceColumnMap.get(new IntWritable(index), columnVector);
      columnVector.times(value).addTo(recommendationVector);
    }

    Queue<RecommendedItem> topItems =
      new PriorityQueue<RecommendedItem>(recommendationsPerUser + 1, Collections.reverseOrder());

    Iterator<Vector.Element> recommendationVectorIterator = recommendationVector.iterateNonZero();
    LongWritable itemID = new LongWritable();
    while (recommendationVectorIterator.hasNext()) {
      Vector.Element element = recommendationVectorIterator.next();
      if (topItems.size() < recommendationsPerUser) {
        indexItemIDMap.get(new IntWritable(element.index()), itemID);
        topItems.add(new GenericRecommendedItem(itemID.get(), (float) element.get()));
      } else if (element.get() > topItems.peek().getValue()) {
        indexItemIDMap.get(new IntWritable(element.index()), itemID);
        topItems.add(new GenericRecommendedItem(itemID.get(), (float) element.get()));
        topItems.poll();
      }
    }

    List<RecommendedItem> recommendations = new ArrayList<RecommendedItem>(topItems.size());
    recommendations.addAll(topItems);
    Collections.sort(recommendations);
    output.collect(userID, new RecommendedItemsWritable(recommendations));
  }

  @Override
  public void close() {
    indexItemIDMap.close();
    cooccurrenceColumnMap.close();
  }

}