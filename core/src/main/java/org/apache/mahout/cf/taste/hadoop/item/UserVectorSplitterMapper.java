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

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.VLongWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.common.FileLineIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public final class UserVectorSplitterMapper extends MapReduceBase implements
    Mapper<VLongWritable,VectorWritable,IntWritable,VectorOrPrefWritable> {

  static final String USERS_FILE = "usersFile";
  
  private FastIDSet usersToRecommendFor;

  @Override
  public void configure(JobConf jobConf) {
    try {
      FileSystem fs = FileSystem.get(jobConf);
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
  }

  @Override
  public void map(VLongWritable key,
                  VectorWritable value,
                  OutputCollector<IntWritable,VectorOrPrefWritable> output,
                  Reporter reporter) throws IOException {
    long userID = key.get();
    if (usersToRecommendFor != null && !usersToRecommendFor.contains(userID)) {
      return;
    }
    Vector userVector = value.get();
    Iterator<Vector.Element> it = userVector.iterateNonZero();
    IntWritable itemIndexWritable = new IntWritable();
    VectorOrPrefWritable vectorOrPref = new VectorOrPrefWritable();
    while (it.hasNext()) {
      Vector.Element e = it.next();
      int itemIndex = e.index();
      double preferenceValue = e.get();
      itemIndexWritable.set(itemIndex);
      vectorOrPref.set(userID, (float) preferenceValue);
      output.collect(itemIndexWritable, vectorOrPref);
    }
  }

}