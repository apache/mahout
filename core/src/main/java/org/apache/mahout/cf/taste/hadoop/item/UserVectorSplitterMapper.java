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

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.lucene.util.PriorityQueue;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class UserVectorSplitterMapper extends
    Mapper<VarLongWritable,VectorWritable, VarIntWritable,VectorOrPrefWritable> {

  private static final Logger log = LoggerFactory.getLogger(UserVectorSplitterMapper.class);

  static final String USERS_FILE = "usersFile";
  static final String MAX_PREFS_PER_USER_CONSIDERED = "maxPrefsPerUserConsidered";
  static final int DEFAULT_MAX_PREFS_PER_USER_CONSIDERED = 10;

  private int maxPrefsPerUserConsidered;
  private FastIDSet usersToRecommendFor;

  private final VarIntWritable itemIndexWritable = new VarIntWritable();
  private final VectorOrPrefWritable vectorOrPref = new VectorOrPrefWritable();

  @Override
  protected void setup(Context context) throws IOException {
    Configuration jobConf = context.getConfiguration();
    maxPrefsPerUserConsidered = jobConf.getInt(MAX_PREFS_PER_USER_CONSIDERED, DEFAULT_MAX_PREFS_PER_USER_CONSIDERED);
    String usersFilePathString = jobConf.get(USERS_FILE);
    if (usersFilePathString != null) {
      FSDataInputStream in = null;
      try {
        Path unqualifiedUsersFilePath = new Path(usersFilePathString);
        FileSystem fs = FileSystem.get(unqualifiedUsersFilePath.toUri(), jobConf);
        usersToRecommendFor = new FastIDSet();
        Path usersFilePath = unqualifiedUsersFilePath.makeQualified(fs);
        in = fs.open(usersFilePath);
        for (String line : new FileLineIterable(in)) {
          try {
            usersToRecommendFor.add(Long.parseLong(line));
          } catch (NumberFormatException nfe) {
            log.warn("usersFile line ignored: {}", line);
          }
        }
      } finally {
        Closeables.close(in, true);
      }
    }
  }

  @Override
  protected void map(VarLongWritable key,
                     VectorWritable value,
                     Context context) throws IOException, InterruptedException {
    long userID = key.get();
    if (usersToRecommendFor != null && !usersToRecommendFor.contains(userID)) {
      return;
    }
    Vector userVector = maybePruneUserVector(value.get());

    for (Element e : userVector.nonZeroes()) {
      itemIndexWritable.set(e.index());
      vectorOrPref.set(userID, (float) e.get());
      context.write(itemIndexWritable, vectorOrPref);
    }
  }

  private Vector maybePruneUserVector(Vector userVector) {
    if (userVector.getNumNondefaultElements() <= maxPrefsPerUserConsidered) {
      return userVector;
    }

    float smallestLargeValue = findSmallestLargeValue(userVector);

    // "Blank out" small-sized prefs to reduce the amount of partial products
    // generated later. They're not zeroed, but NaN-ed, so they come through
    // and can be used to exclude these items from prefs.
    for (Element e : userVector.nonZeroes()) {
      float absValue = Math.abs((float) e.get());
      if (absValue < smallestLargeValue) {
        e.set(Float.NaN);
      }
    }

    return userVector;
  }

  private float findSmallestLargeValue(Vector userVector) {

    PriorityQueue<Float> topPrefValues = new PriorityQueue<Float>(maxPrefsPerUserConsidered) {
      @Override
      protected boolean lessThan(Float f1, Float f2) {
        return f1 < f2;
      }
    };

    for (Element e : userVector.nonZeroes()) {
      float absValue = Math.abs((float) e.get());
      topPrefValues.insertWithOverflow(absValue);
    }
    return topPrefValues.top();
  }

}
