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
import java.util.Collections;
import java.util.Iterator;
import java.util.PriorityQueue;

import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.map.OpenIntIntHashMap;

public final class UserVectorToCooccurrenceMapper extends
    Mapper<VarLongWritable,VectorWritable,VarIntWritable,VarIntWritable> {

  private static final int MAX_PREFS_CONSIDERED = 100;

  private enum Counters {
    USER_PREFS_SKIPPED,
  }

  private final OpenIntIntHashMap indexCounts = new OpenIntIntHashMap();

  @Override
  public void map(VarLongWritable userID,
                  VectorWritable userVectorWritable,
                  Context context) throws IOException, InterruptedException {

    Vector userVector = userVectorWritable.get();
    countSeen(userVector);

    int originalSize = userVector.getNumNondefaultElements();
    userVector = maybePruneUserVector(userVector);
    int newSize = userVector.getNumNondefaultElements();
    if (newSize < originalSize) {
      context.getCounter(Counters.USER_PREFS_SKIPPED).increment(originalSize - newSize);
    }

    Iterator<Vector.Element> it = userVector.iterateNonZero();
    VarIntWritable itemIndex1 = new VarIntWritable();
    VarIntWritable itemIndex2 = new VarIntWritable();
    while (it.hasNext()) {
      int index1 = it.next().index();
      itemIndex1.set(index1);
      Iterator<Vector.Element> it2 = userVector.iterateNonZero();
      while (it2.hasNext()) {
        int index2 = it2.next().index();
        itemIndex2.set(index2);
        context.write(itemIndex1, itemIndex2);
      }
    }
  }

  private Vector maybePruneUserVector(Vector userVector) {
    if (userVector.getNumNondefaultElements() <= MAX_PREFS_CONSIDERED) {
      return userVector;
    }

    PriorityQueue<Integer> smallCounts =
        new PriorityQueue<Integer>(MAX_PREFS_CONSIDERED + 1, Collections.reverseOrder());

    Iterator<Vector.Element> it = userVector.iterateNonZero();
    while (it.hasNext()) {
      int count = indexCounts.get(it.next().index());
      if (count > 0) {
        if (smallCounts.size() < MAX_PREFS_CONSIDERED) {
          smallCounts.add(count);
        } else if (count < smallCounts.peek()) {
          smallCounts.add(count);
          smallCounts.poll();
        }
      }
    }
    int greatestSmallCount = smallCounts.peek();

    if (greatestSmallCount > 0) {
      Iterator<Vector.Element> it2 = userVector.iterateNonZero();
      while (it2.hasNext()) {
        Vector.Element e = it2.next();
        if (indexCounts.get(e.index()) > greatestSmallCount) {
          e.set(0.0);
        }
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
