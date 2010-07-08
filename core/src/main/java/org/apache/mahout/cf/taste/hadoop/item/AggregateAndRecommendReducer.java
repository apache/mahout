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
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.recommender.ByValueRecommendedItemComparator;
import org.apache.mahout.cf.taste.impl.recommender.GenericRecommendedItem;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.common.FileLineIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.function.UnaryFunction;
import org.apache.mahout.math.map.OpenIntLongHashMap;

/**
 * <p>computes prediction values for each user</p>
 *
 * <pre>
 * u = a user
 * i = an item not yet rated by u
 * N = all items similar to i (where similarity is usually computed by pairwisely comparing the item-vectors
 * of the user-item matrix)
 *
 * Prediction(u,i) = sum(all n from N: similarity(i,n) * rating(u,n)) / sum(all n from N: abs(similarity(i,n)))
 * </pre>
 */
public final class AggregateAndRecommendReducer extends
    Reducer<VarLongWritable,PrefAndSimilarityColumnWritable,VarLongWritable,RecommendedItemsWritable> {

  static final String ITEMID_INDEX_PATH = "itemIDIndexPath";
  static final String NUM_RECOMMENDATIONS = "numRecommendations";
  static final int DEFAULT_NUM_RECOMMENDATIONS = 10;
  static final String ITEMS_FILE = "itemsFile";

  private boolean booleanData;
  private int recommendationsPerUser;
  private FastIDSet itemsToRecommendFor;
  private OpenIntLongHashMap indexItemIDMap;

  @Override
  protected void setup(Context context) {
    Configuration jobConf = context.getConfiguration();
    recommendationsPerUser = jobConf.getInt(NUM_RECOMMENDATIONS, DEFAULT_NUM_RECOMMENDATIONS);
    booleanData = jobConf.getBoolean(RecommenderJob.BOOLEAN_DATA, false);
    try {
      FileSystem fs = FileSystem.get(jobConf);
      Path itemIDIndexPath = new Path(jobConf.get(ITEMID_INDEX_PATH)).makeQualified(fs);
      indexItemIDMap = new OpenIntLongHashMap();
      VarIntWritable index = new VarIntWritable();
      VarLongWritable id = new VarLongWritable();
      for (FileStatus status : fs.listStatus(itemIDIndexPath, TasteHadoopUtils.PARTS_FILTER)) {
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
        String itemFilePathString = jobConf.get(ITEMS_FILE);
        if (itemFilePathString == null) {
        	itemsToRecommendFor = null;
        } else {
        	itemsToRecommendFor = new FastIDSet();
          Path usersFilePath = new Path(itemFilePathString).makeQualified(fs);
          FSDataInputStream in = fs.open(usersFilePath);
          for (String line : new FileLineIterable(in)) {
        	  itemsToRecommendFor.add(Long.parseLong(line));
          }
        }
      } catch (IOException ioe) {
        throw new IllegalStateException(ioe);
      }
  }

  private static final UnaryFunction ABSOLUTE_VALUES = new UnaryFunction() {
    @Override
    public double apply(double value) {
      return value < 0 ? value * -1 : value;
    }
  };

  @Override
  protected void reduce(VarLongWritable userID,
                        Iterable<PrefAndSimilarityColumnWritable> values,
                        Context context) throws IOException, InterruptedException {
    if (booleanData) {
      reduceBooleanData(userID, values, context);
    } else {
      reduceNonBooleanData(userID, values, context);
    }
  }

  private void reduceBooleanData(VarLongWritable userID,
                        Iterable<PrefAndSimilarityColumnWritable> values,
                        Context context) throws IOException, InterruptedException {

    /* having boolean data, each estimated preference can only be 1,
     * so the computation is much simpler */
    Vector predictionVector = null;
    for (PrefAndSimilarityColumnWritable prefAndSimilarityColumn : values) {
      predictionVector = predictionVector == null
          ? prefAndSimilarityColumn.getSimilarityColumn()
          : predictionVector.plus(prefAndSimilarityColumn.getSimilarityColumn());
    }

    Iterator<Element> predictions = predictionVector.iterateNonZero();
    List<RecommendedItem> recommendations = new ArrayList<RecommendedItem>();
    while (predictions.hasNext() && recommendations.size() < recommendationsPerUser) {
      int itemIDIndex = predictions.next().index();
      long itemID = indexItemIDMap.get(itemIDIndex);
      if (itemsToRecommendFor == null || itemsToRecommendFor.contains(itemID)) {
        recommendations.add(new GenericRecommendedItem(itemID, 1f));
      }
    }

    if (!recommendations.isEmpty()) {
      context.write(userID, new RecommendedItemsWritable(recommendations));
    }
  }

  private void reduceNonBooleanData(VarLongWritable userID,
                        Iterable<PrefAndSimilarityColumnWritable> values,
                        Context context) throws IOException, InterruptedException {
    /* each entry here is the sum in the numerator of the prediction formula */
    Vector numerators = null;
    /* each entry here is the sum in the denominator of the prediction formula */
    Vector denominators = null;
    /* each entry here is the number of similar items used in the prediction formula */
    Vector numberOfSimilarItemsUsed = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);

    for (PrefAndSimilarityColumnWritable prefAndSimilarityColumn : values) {
      Vector simColumn = prefAndSimilarityColumn.getSimilarityColumn();
      float prefValue = prefAndSimilarityColumn.getPrefValue();
      /* count the number of items used for each prediction */
      Iterator<Element> usedItemsIterator = simColumn.iterateNonZero();
      while (usedItemsIterator.hasNext()) {
        int itemIDIndex = usedItemsIterator.next().index();
        numberOfSimilarItemsUsed.setQuick(itemIDIndex, numberOfSimilarItemsUsed.getQuick(itemIDIndex) + 1);
      }

      numerators = numerators == null
          ? prefValue == 1.0f ? simColumn.clone() : simColumn.times(prefValue)
          : numerators.plus(prefValue == 1.0f ? simColumn : simColumn.times(prefValue));

      simColumn.assign(ABSOLUTE_VALUES);
      denominators = denominators == null ? simColumn : denominators.plus(simColumn);
    }

    if (numerators == null) {
      return;
    }

    Vector recommendationVector = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    Iterator<Element> iterator = numerators.iterateNonZero();
    while (iterator.hasNext()) {
      Element element = iterator.next();
      int itemIDIndex = element.index();
      /* preference estimations must be based on at least 2 datapoints */
      if (numberOfSimilarItemsUsed.getQuick(itemIDIndex) > 1) {
        /* compute normalized prediction */
        double prediction = element.get() / denominators.getQuick(itemIDIndex);
        recommendationVector.setQuick(itemIDIndex, prediction);
      }
    }

    Queue<RecommendedItem> topItems = new PriorityQueue<RecommendedItem>(recommendationsPerUser + 1,
    Collections.reverseOrder(ByValueRecommendedItemComparator.getInstance()));

    Iterator<Vector.Element> recommendationVectorIterator = recommendationVector.iterateNonZero();
    while (recommendationVectorIterator.hasNext()) {
      Vector.Element element = recommendationVectorIterator.next();
      int index = element.index();

      long itemID = indexItemIDMap.get(index);
      if (itemsToRecommendFor == null || itemsToRecommendFor.contains(itemID)) {
        float value = (float) element.get();
        if (!Float.isNaN(value)) {
          if (topItems.size() < recommendationsPerUser) {
            topItems.add(new GenericRecommendedItem(itemID, value));
          } else if (value > topItems.peek().getValue()) {
            topItems.add(new GenericRecommendedItem(itemID, value));
            topItems.poll();
          }
        }
      }
    }

    if (!topItems.isEmpty()) {
      List<RecommendedItem> recommendations = new ArrayList<RecommendedItem>(topItems.size());
      recommendations.addAll(topItems);
      Collections.sort(recommendations, ByValueRecommendedItemComparator.getInstance());
      context.write(userID, new RecommendedItemsWritable(recommendations));
    }
  }

}
