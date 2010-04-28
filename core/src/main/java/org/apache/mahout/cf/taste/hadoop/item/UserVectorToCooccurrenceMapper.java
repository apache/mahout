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
import java.util.Iterator;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.map.OpenIntIntHashMap;

public final class UserVectorToCooccurrenceMapper extends MapReduceBase implements
    Mapper<LongWritable, VectorWritable,IndexIndexWritable,IntWritable> {

  private static final int MAX_PREFS_CONSIDERED = 50;

  private boolean outputGuardValue = true;
  private final OpenIntIntHashMap indexCounts = new OpenIntIntHashMap();

  @Override
  public void map(LongWritable userID,
                  VectorWritable userVectorWritable,
                  OutputCollector<IndexIndexWritable,IntWritable> output,
                  Reporter reporter) throws IOException {
    Vector userVector = maybePruneUserVector(userVectorWritable.get());
    countSeen(userVector);
    Iterator<Vector.Element> it = userVector.iterateNonZero();
    IndexIndexWritable entityEntity = new IndexIndexWritable();
    IntWritable one = new IntWritable(1);
    while (it.hasNext()) {
      int index1 = it.next().index();
      Iterator<Vector.Element> it2 = userVector.iterateNonZero();
      while (it2.hasNext()) {
        int index2 = it2.next().index();
        if (index1 != index2) {
          entityEntity.set(index1, index2);
          output.collect(entityEntity, one);
        }
      }
    }
    // Guard value, output once, sorts after everything; will be dropped by combiner
    if (outputGuardValue) {
      output.collect(new IndexIndexWritable(Integer.MAX_VALUE, Integer.MAX_VALUE), one);
      outputGuardValue = false;
    }
  }

  private Vector maybePruneUserVector(Vector userVector) {
    if (userVector.getNumNondefaultElements() <= MAX_PREFS_CONSIDERED) {
      return userVector;
    }

    OpenIntIntHashMap countCounts = new OpenIntIntHashMap();
    Iterator<Vector.Element> it = userVector.iterateNonZero();
    while (it.hasNext()) {
      int index = it.next().index();
      int count = indexCounts.get(index);
      countCounts.adjustOrPutValue(count, 1, 1);
    }

    int resultingSizeAtCutoff = 0;
    int cutoff = 0;
    while (resultingSizeAtCutoff <= MAX_PREFS_CONSIDERED) {
      cutoff++;
      int count = indexCounts.get(cutoff);
      resultingSizeAtCutoff += count;
    }

    Iterator<Vector.Element> it2 = userVector.iterateNonZero();
    while (it2.hasNext()) {
      Vector.Element e = it2.next();
      int index = e.index();
      int count = indexCounts.get(index);
      if (count > cutoff) {
        e.set(0.0);
      }
    }

    return userVector;
  }

  private void countSeen(Vector userVector) {
    Iterator<Vector.Element> it = userVector.iterateNonZero();
    while (it.hasNext()) {
      int index = it.next().index();
      indexCounts.adjustOrPutValue(index, 1, 1);
    }
  }
  
}
