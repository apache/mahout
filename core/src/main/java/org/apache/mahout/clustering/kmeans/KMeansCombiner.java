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

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class KMeansCombiner extends Reducer<Text, KMeansInfo, Text, KMeansInfo> {

  /* (non-Javadoc)
   * @see org.apache.hadoop.mapreduce.Reducer#reduce(java.lang.Object, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
   */
  @Override
  protected void reduce(Text key, Iterable<KMeansInfo> values, Context context) throws IOException, InterruptedException {

    Cluster cluster = new Cluster(key.toString());
    Iterator<KMeansInfo> it = values.iterator();
    while (it.hasNext()) {
      KMeansInfo next = it.next();
      cluster.addPoints(next.getPoints(), next.getPointTotal());
    }
    context.write(key, new KMeansInfo(cluster.getNumPoints(), cluster.getPointTotal()));
  }

}
