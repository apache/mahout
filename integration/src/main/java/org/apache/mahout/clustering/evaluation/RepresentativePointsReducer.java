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

package org.apache.mahout.clustering.evaluation;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.clustering.classify.WeightedVectorWritable;
import org.apache.mahout.math.VectorWritable;

public class RepresentativePointsReducer
  extends Reducer<IntWritable, WeightedVectorWritable, IntWritable, VectorWritable> {

  private Map<Integer, List<VectorWritable>> representativePoints;

  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    for (Map.Entry<Integer, List<VectorWritable>> entry : representativePoints.entrySet()) {
      IntWritable iw = new IntWritable(entry.getKey());
      for (VectorWritable vw : entry.getValue()) {
        context.write(iw, vw);
      }
    }
    super.cleanup(context);
  }

  @Override
  protected void reduce(IntWritable key, Iterable<WeightedVectorWritable> values, Context context)
    throws IOException, InterruptedException {
    // find the most distant point
    WeightedVectorWritable mdp = null;
    for (WeightedVectorWritable dpw : values) {
      if (mdp == null || mdp.getWeight() < dpw.getWeight()) {
        mdp = new WeightedVectorWritable(dpw.getWeight(), dpw.getVector());
      }
    }
    context.write(new IntWritable(key.get()), new VectorWritable(mdp.getVector()));
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    representativePoints = RepresentativePointsMapper.getRepresentativePoints(conf);
  }

  public void configure(Map<Integer, List<VectorWritable>> representativePoints) {
    this.representativePoints = representativePoints;
  }

}
