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

package org.apache.mahout.clustering.fuzzykmeans;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class FuzzyKMeansCombiner extends Reducer<Text,FuzzyKMeansInfo,Text,FuzzyKMeansInfo> {
  
  private FuzzyKMeansClusterer clusterer;
  
  
  /* (non-Javadoc)
   * @see org.apache.hadoop.mapreduce.Reducer#reduce(java.lang.Object, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
   */
  @Override
  protected void reduce(Text key, Iterable<FuzzyKMeansInfo> values, Context context) throws IOException, InterruptedException {
    SoftCluster cluster = new SoftCluster(key.toString().trim());
    Iterator<FuzzyKMeansInfo> it = values.iterator();
    while (it.hasNext()) {
      FuzzyKMeansInfo info = it.next();   
      if (info.getCombinerPass() == 0) { // first time thru combiner
        cluster.addPoint(info.getVector(), Math.pow(info.getProbability(), clusterer.getM()));
      } else {
        cluster.addPoints(info.getVector(), info.getProbability());
      }
      info.setCombinerPass(info.getCombinerPass() + 1);
    }
    // TODO: how do we pass along the combinerPass? Or do we not need to?
    context.write(key, new FuzzyKMeansInfo(cluster.getPointProbSum(), cluster.getWeightedPointTotal(), 1));
    }

  /* (non-Javadoc)
   * @see org.apache.hadoop.mapreduce.Reducer#setup(org.apache.hadoop.mapreduce.Reducer.Context)
   */
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    clusterer = new FuzzyKMeansClusterer(context.getConfiguration());
  }
}
