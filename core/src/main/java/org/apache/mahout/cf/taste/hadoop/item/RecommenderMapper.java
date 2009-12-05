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
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
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
    implements Mapper<LongWritable, Vector, LongWritable, RecommendedItemsWritable> {

  static final String COOCCURRENCE_PATH = "cooccurrencePath";
  static final String RECOMMENDATIONS_PER_USER = "recommendationsPerUser";

  private FileSystem fs;
  private Path cooccurrencePath;
  private int recommendationsPerUser;

  @Override
  public void configure(JobConf jobConf) {
    try {
      fs = FileSystem.get(jobConf);
    } catch (IOException ioe) {
      throw new IllegalStateException(ioe);
    }
    cooccurrencePath = new Path(jobConf.get(COOCCURRENCE_PATH)).makeQualified(fs);
    recommendationsPerUser = jobConf.getInt(RECOMMENDATIONS_PER_USER, 10);
  }

  @Override
  public void map(LongWritable userID,
                  Vector userVector,
                  OutputCollector<LongWritable, RecommendedItemsWritable> output,
                  Reporter reporter) throws IOException {

    SequenceFile.Reader reader = new SequenceFile.Reader(fs, cooccurrencePath, new Configuration());
    LongWritable itemIDWritable = new LongWritable();
    Vector cooccurrenceVector = new SparseVector();
    Queue<RecommendedItem> topItems =
        new PriorityQueue<RecommendedItem>(recommendationsPerUser + 1, Collections.reverseOrder());
    while (reader.next(itemIDWritable, cooccurrenceVector)) {
      processOneRecommendation(userVector, itemIDWritable.get(), cooccurrenceVector, topItems);
    }
    List<RecommendedItem> recommendations = new ArrayList<RecommendedItem>(topItems.size());
    recommendations.addAll(topItems);
    output.collect(userID, new RecommendedItemsWritable(recommendations));
  }

  private void processOneRecommendation(Vector userVector,
                                        long itemID,
                                        Vector cooccurrenceVector,
                                        Queue<RecommendedItem> topItems) {
    double totalWeight = 0.0;
    Iterator<Vector.Element> cooccurrences = cooccurrenceVector.iterateNonZero();
    while (cooccurrences.hasNext()) {
      Vector.Element cooccurrence = cooccurrences.next();
      totalWeight += cooccurrence.get();
    }
    double score = userVector.dot(cooccurrenceVector) / totalWeight;
    if (!Double.isNaN(score)) {
      topItems.add(new GenericRecommendedItem(itemID, (float) score));
      if (topItems.size() > recommendationsPerUser) {
        topItems.poll();
      }
    }
  }

}