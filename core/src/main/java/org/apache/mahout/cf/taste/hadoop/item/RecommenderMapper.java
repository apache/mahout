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
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.hadoop.MapFilesMap;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.impl.common.Cache;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.Retriever;
import org.apache.mahout.cf.taste.impl.recommender.ByValueRecommendedItemComparator;
import org.apache.mahout.cf.taste.impl.recommender.GenericRecommendedItem;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.common.FileLineIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public final class RecommenderMapper extends MapReduceBase implements
    Mapper<LongWritable,VectorWritable,LongWritable,RecommendedItemsWritable> {
  
  static final String COOCCURRENCE_PATH = "cooccurrencePath";
  static final String ITEMID_INDEX_PATH = "itemIDIndexPath";
  static final String RECOMMENDATIONS_PER_USER = "recommendationsPerUser";
  static final String USERS_FILE = "usersFile";
  
  private int recommendationsPerUser;
  private MapFilesMap<IntWritable,LongWritable> indexItemIDMap;
  private MapFilesMap<IntWritable,VectorWritable> cooccurrenceColumnMap;
  private Cache<IntWritable,Vector> cooccurrenceColumnCache;
  private FastIDSet usersToRecommendFor;
  private boolean booleanData;
  
  @Override
  public void configure(JobConf jobConf) {
    try {
      FileSystem fs = FileSystem.get(jobConf);
      Path cooccurrencePath = new Path(jobConf.get(COOCCURRENCE_PATH)).makeQualified(fs);
      Path itemIDIndexPath = new Path(jobConf.get(ITEMID_INDEX_PATH)).makeQualified(fs);
      recommendationsPerUser = jobConf.getInt(RECOMMENDATIONS_PER_USER, 10);
      indexItemIDMap = new MapFilesMap<IntWritable,LongWritable>(fs, itemIDIndexPath, new Configuration());
      cooccurrenceColumnMap = new MapFilesMap<IntWritable,VectorWritable>(fs, cooccurrencePath,
          new Configuration());
      String usersFilePathString = jobConf.get(USERS_FILE);
      if (usersFilePathString == null) {
        usersToRecommendFor = null;
      } else {
        usersToRecommendFor = new FastIDSet();
        Path usersFilePath = new Path(usersFilePathString).makeQualified(fs);
        FSDataInputStream in = fs.open(usersFilePath);
        for (String line : new FileLineIterable(in)) {
          usersToRecommendFor.add(Long.parseLong(line));
        }
      }
    } catch (IOException ioe) {
      throw new IllegalStateException(ioe);
    }
    cooccurrenceColumnCache = new Cache<IntWritable,Vector>(new CooccurrenceCache(cooccurrenceColumnMap), 100);
    booleanData = jobConf.getBoolean(RecommenderJob.BOOLEAN_DATA, false);
  }
  
  @Override
  public void map(LongWritable userID,
                  VectorWritable vectorWritable,
                  OutputCollector<LongWritable,RecommendedItemsWritable> output,
                  Reporter reporter) throws IOException {
    
    if ((usersToRecommendFor != null) && !usersToRecommendFor.contains(userID.get())) {
      return;
    }
    Vector userVector = vectorWritable.get();
    Iterator<Vector.Element> userVectorIterator = userVector.iterateNonZero();
    Vector recommendationVector = new RandomAccessSparseVector(Integer.MAX_VALUE, 1000);
    while (userVectorIterator.hasNext()) {
      Vector.Element element = userVectorIterator.next();
      int index = element.index();
      Vector columnVector;
      try {
        columnVector = cooccurrenceColumnCache.get(new IntWritable(index));
      } catch (TasteException te) {
        if (te.getCause() instanceof IOException) {
          throw (IOException) te.getCause();
        } else {
          throw new IOException(te.getCause());
        }
      }
      if (columnVector != null) {
        if (booleanData) { // because 'value' is 1.0
          columnVector.addTo(recommendationVector);
        } else {
          double value = element.get();          
          columnVector.times(value).addTo(recommendationVector);
        }
      }
    }
    
    Queue<RecommendedItem> topItems = new PriorityQueue<RecommendedItem>(recommendationsPerUser + 1,
        Collections.reverseOrder(ByValueRecommendedItemComparator.getInstance()));
    
    Iterator<Vector.Element> recommendationVectorIterator = recommendationVector.iterateNonZero();
    LongWritable itemID = new LongWritable();
    while (recommendationVectorIterator.hasNext()) {
      Vector.Element element = recommendationVectorIterator.next();
      int index = element.index();
      if (userVector.get(index) == 0.0) {
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
    }

    if (!topItems.isEmpty()) {
      List<RecommendedItem> recommendations = new ArrayList<RecommendedItem>(topItems.size());
      recommendations.addAll(topItems);
      Collections.sort(recommendations, ByValueRecommendedItemComparator.getInstance());
      output.collect(userID, new RecommendedItemsWritable(recommendations));
    }
  }
  
  @Override
  public void close() {
    indexItemIDMap.close();
    cooccurrenceColumnMap.close();
  }
  
  private static class CooccurrenceCache implements Retriever<IntWritable,Vector> {
    
    private final MapFilesMap<IntWritable,VectorWritable> map;

    private CooccurrenceCache(MapFilesMap<IntWritable,VectorWritable> map) {
      this.map = map;
    }
    
    @Override
    public Vector get(IntWritable key) throws TasteException {
      VectorWritable columnVector = new VectorWritable();
      try {
        columnVector = map.get(key, columnVector);
      } catch (IOException ioe) {
        throw new TasteException(ioe);
      }
      if (columnVector == null) {
        return null;
      }
      Vector value = columnVector.get();
      if (value == null) {
        throw new IllegalStateException("Vector in map file was empty?");
      }
      return value;
    }
    
  }
}