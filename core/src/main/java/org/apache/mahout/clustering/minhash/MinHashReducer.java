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

package org.apache.mahout.clustering.minhash;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.common.commandline.MinhashOptionCreator;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.Collection;

public class MinHashReducer extends Reducer<Text,Writable,Text,Writable> {
  
  private int minClusterSize;
  private boolean debugOutput;
  
  enum Clusters {
    ACCEPTED,
    DISCARDED
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    this.minClusterSize = conf.getInt(MinhashOptionCreator.MIN_CLUSTER_SIZE, 5);
    this.debugOutput = conf.getBoolean(MinhashOptionCreator.DEBUG_OUTPUT, false);
  }
  
  /**
   * output the items clustered
   */
  @Override
  protected void reduce(Text cluster, Iterable<Writable> points, Context context)
    throws IOException, InterruptedException {
    Collection<Writable> pointList = Lists.newArrayList();
    for (Writable point : points) {
      if (debugOutput) {
        Vector pointVector = ((VectorWritable) point).get().clone();
        Writable writablePointVector = new VectorWritable(pointVector);
        pointList.add(writablePointVector);
      } else {
        Writable pointText = new Text(point.toString());
        pointList.add(pointText);
      }
    }
    if (pointList.size() >= minClusterSize) {
      context.getCounter(Clusters.ACCEPTED).increment(1);
      for (Writable point : pointList) {
        context.write(cluster, point);
      }
    } else {
      context.getCounter(Clusters.DISCARDED).increment(1);
    }
  }
  
}
