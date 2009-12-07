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
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
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

  private static final PathFilter IGNORABLE_FILES_FILTER = new PathFilter() {
    @Override
    public boolean accept(Path path) {
      return !path.getName().startsWith("_logs");
    }
  };

  private FileSystem fs;
  private Path cooccurrencePath;
  private int recommendationsPerUser;
  private FastByIDMap<Long> indexItemIDMap;

  @Override
  public void configure(JobConf jobConf) {
    try {
      fs = FileSystem.get(jobConf);
    } catch (IOException ioe) {
      throw new IllegalStateException(ioe);
    }
    cooccurrencePath = new Path(jobConf.get(COOCCURRENCE_PATH)).makeQualified(fs);
    Path itemIDIndexPath = new Path(jobConf.get(ITEMID_INDEX_PATH)).makeQualified(fs);
    recommendationsPerUser = jobConf.getInt(RECOMMENDATIONS_PER_USER, 10);
    indexItemIDMap = new FastByIDMap<Long>();
    try {
      IntWritable index = new IntWritable();
      LongWritable itemID = new LongWritable();
      Configuration conf = new Configuration();
      for (FileStatus status : fs.listStatus(itemIDIndexPath, IGNORABLE_FILES_FILTER)) {
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, status.getPath(), conf);
        while (reader.next(index, itemID)) {
          indexItemIDMap.put(index.get(), itemID.get());
        }
        reader.close();
      }
    } catch (IOException ioe) {
      throw new IllegalStateException(ioe);
    }
  }

  @Override
  public void map(LongWritable userID,
                  SparseVector userVector,
                  OutputCollector<LongWritable, RecommendedItemsWritable> output,
                  Reporter reporter) throws IOException {
    IntWritable indexWritable = new IntWritable();
    Vector cooccurrenceVector = new SparseVector(Integer.MAX_VALUE, 1000);
    Configuration conf = new Configuration();
    Queue<RecommendedItem> topItems =
        new PriorityQueue<RecommendedItem>(recommendationsPerUser + 1, Collections.reverseOrder());
    for (FileStatus status : fs.listStatus(cooccurrencePath, IGNORABLE_FILES_FILTER)) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, status.getPath(), conf);
      while (reader.next(indexWritable, cooccurrenceVector)) {
        Long itemID = indexItemIDMap.get(indexWritable.get());
        if (itemID != null) {
          processOneRecommendation(userVector, itemID, cooccurrenceVector, topItems);
        } else {
          throw new IllegalStateException("Found index without item ID: " + indexWritable.get());
        }
      }
      reader.close();
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