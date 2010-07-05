/*
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
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.recommender.ByValueRecommendedItemComparator;
import org.apache.mahout.cf.taste.impl.recommender.GenericRecommendedItem;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.common.FileLineIterable;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.map.OpenIntLongHashMap;

public final class AggregateAndRecommendReducer extends
    Reducer<VarLongWritable,VectorWritable,VarLongWritable,RecommendedItemsWritable> {

  static final String ITEMID_INDEX_PATH = "itemIDIndexPath";
  static final String NUM_RECOMMENDATIONS = "numRecommendations";
  static final int DEFAULT_NUM_RECOMMENDATIONS = 10;
  static final String ITEMS_FILE="itemsFile";
  
  private FastIDSet itemsToRecommendFor;
  
  private static final PathFilter PARTS_FILTER = new PathFilter() {
    @Override
    public boolean accept(Path path) {
      return path.getName().startsWith("part-");
    }
  };

  private int recommendationsPerUser;
  private OpenIntLongHashMap indexItemIDMap;

  @Override
  protected void setup(Context context) {
    Configuration jobConf = context.getConfiguration();
    recommendationsPerUser = jobConf.getInt(NUM_RECOMMENDATIONS, DEFAULT_NUM_RECOMMENDATIONS);
    try {
      FileSystem fs = FileSystem.get(jobConf);
      Path itemIDIndexPath = new Path(jobConf.get(ITEMID_INDEX_PATH)).makeQualified(fs);
      indexItemIDMap = new OpenIntLongHashMap();
      VarIntWritable index = new VarIntWritable();
      VarLongWritable id = new VarLongWritable();
      for (FileStatus status : fs.listStatus(itemIDIndexPath, PARTS_FILTER)) {
        String path = status.getPath().toString();
        SequenceFile.Reader reader =
            new SequenceFile.Reader(fs, new Path(path).makeQualified(fs), jobConf);
        while (reader.next(index, id)) {
          indexItemIDMap.put(index.get(), id.get());
        }
        reader.close();
      }
    } catch (IOException ioe) {
      throw new IllegalStateException(ioe);
    }
    
    try {
        FileSystem fs = FileSystem.get(jobConf);
        String usersFilePathString = jobConf.get(ITEMS_FILE);
        if (usersFilePathString == null) {
        	itemsToRecommendFor = null;
        } else {
        	itemsToRecommendFor = new FastIDSet();
          Path usersFilePath = new Path(usersFilePathString).makeQualified(fs);
          FSDataInputStream in = fs.open(usersFilePath);
          for (String line : new FileLineIterable(in)) {
        	  itemsToRecommendFor.add(Long.parseLong(line));
          }
        }
      } catch (IOException ioe) {
        throw new IllegalStateException(ioe);
      }
  }

  @Override
  protected void reduce(VarLongWritable key,
                        Iterable<VectorWritable> values,
                        Context context) throws IOException, InterruptedException {

    Vector recommendationVector = null;
    for (VectorWritable vectorWritable : values) {
      recommendationVector = recommendationVector == null
          ? vectorWritable.get()
          : recommendationVector.plus(vectorWritable.get());
    }
    if (recommendationVector == null) {
      return;
    }

    Queue<RecommendedItem> topItems = new PriorityQueue<RecommendedItem>(recommendationsPerUser + 1,
    Collections.reverseOrder(ByValueRecommendedItemComparator.getInstance()));

    Iterator<Vector.Element> recommendationVectorIterator =
        recommendationVector.iterateNonZero();
    while (recommendationVectorIterator.hasNext()) {
      Vector.Element element = recommendationVectorIterator.next();
      int index = element.index();
  
      long itemId = indexItemIDMap.get(index);
      if (itemsToRecommendFor == null || itemsToRecommendFor.contains(itemId)) {
        float value = (float) element.get();
        if (!Float.isNaN(value)) {
          if (topItems.size() < recommendationsPerUser) {
            topItems.add(new GenericRecommendedItem(itemId, value));
          } else if (value > topItems.peek().getValue()) {
            topItems.add(new GenericRecommendedItem(itemId, value));
            topItems.poll();
          }
        }
      }
    }

    if (!topItems.isEmpty()) {
      List<RecommendedItem> recommendations = new ArrayList<RecommendedItem>(topItems.size());
      recommendations.addAll(topItems);
      Collections.sort(recommendations, ByValueRecommendedItemComparator.getInstance());
      context.write(key, new RecommendedItemsWritable(recommendations));
    }
  }

}
