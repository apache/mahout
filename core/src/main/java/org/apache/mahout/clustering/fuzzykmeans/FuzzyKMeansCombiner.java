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

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class FuzzyKMeansCombiner extends Reducer<Text,FuzzyKMeansInfo,Text,FuzzyKMeansInfo> {
  
  private FuzzyKMeansClusterer clusterer;

  @Override
  protected void reduce(Text key, Iterable<FuzzyKMeansInfo> values, Context context) throws IOException, InterruptedException {
    SoftCluster cluster = new SoftCluster(key.toString().trim());
    for (FuzzyKMeansInfo value : values) {
      if (value.getCombinerPass() == 0) { // first time thru combiner
        cluster.addPoint(value.getVector(), Math.pow(value.getProbability(), clusterer.getM()));
      } else {
        cluster.addPoints(value.getVector(), value.getProbability());
      }
      value.setCombinerPass(value.getCombinerPass() + 1);
    }
    context.write(key, new FuzzyKMeansInfo(cluster.getPointProbSum(), cluster.getWeightedPointTotal(), 1));
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    clusterer = new FuzzyKMeansClusterer(context.getConfiguration());
  }
}
