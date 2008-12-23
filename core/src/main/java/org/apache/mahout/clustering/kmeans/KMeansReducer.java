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

package org.apache.mahout.clustering.kmeans;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.matrix.AbstractVector;
import org.apache.mahout.matrix.Vector;

import java.io.IOException;
import java.util.Iterator;

public class KMeansReducer extends MapReduceBase implements
        Reducer<Text, Text, Text, Text> {

  //double delta = 0;

  @Override
  public void reduce(Text key, Iterator<Text> values,
                     OutputCollector<Text, Text> output, Reporter reporter) throws IOException {
    Cluster cluster = Cluster.decodeCluster(key.toString());
    while (values.hasNext()) {
      String value = values.next().toString();
      int ix = value.indexOf(',');
      int count = Integer.parseInt(value.substring(0, ix));
      Vector total = AbstractVector.decodeVector(value.substring(ix + 2));
      cluster.addPoints(count, total);
    }
    // force convergence calculation
    cluster.computeConvergence();
    output.collect(new Text(cluster.getIdentifier()), new Text(Cluster
            .formatCluster(cluster)));
  }

  @Override
  public void configure(JobConf job) {
    super.configure(job);
    Cluster.configure(job);
  }

}
