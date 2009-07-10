/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
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

import java.io.IOException;
import java.util.Iterator;

public class KMeansCombiner extends MapReduceBase implements
    Reducer<Text, KMeansInfo, Text, KMeansInfo> {

  @Override
  public void reduce(Text key, Iterator<KMeansInfo> values,
                     OutputCollector<Text, KMeansInfo> output, Reporter reporter) throws IOException {
    Cluster cluster = new Cluster(key.toString());
    while (values.hasNext()) {
      KMeansInfo next = values.next();
      cluster.addPoints(next.getPoints(),
          next.getPointTotal());
    }
    output.collect(key, new KMeansInfo(cluster.getNumPoints(), cluster.getPointTotal()));
  }

  @Override
  public void configure(JobConf job) {
    super.configure(job);
    Cluster.configure(job);
  }

}
