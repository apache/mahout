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
package org.apache.mahout.clustering.canopy;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.lib.IdentityReducer;

public class ClusterDriver {

  public static void main(String[] args) {
    String points = args[0];
    String canopies = args[1];
    String output = args[2];
    String measureClassName = args[3];
    float t1 = new Float(args[4]);
    float t2 = new Float(args[5]);
    runJob(points, canopies, output, measureClassName, t1, t2);
  }

  /**
   * Run the job
   *
   * @param points           the input points directory pathname String
   * @param canopies         the input canopies directory pathname String
   * @param output           the output directory pathname String
   * @param measureClassName the DistanceMeasure class name
   * @param t1               the T1 distance threshold
   * @param t2               the T2 distance threshold
   * @param jarLocation
   */
  public static void runJob(String points, String canopies, String output,
                            String measureClassName, float t1, float t2) {
    JobClient client = new JobClient();
    JobConf conf = new JobConf(
            org.apache.mahout.clustering.canopy.ClusterDriver.class);

    conf.set(Canopy.DISTANCE_MEASURE_KEY, measureClassName);
    conf.set(Canopy.T1_KEY, "" + t1);
    conf.set(Canopy.T2_KEY, "" + t2);
    conf.set(Canopy.CANOPY_PATH_KEY, canopies);

    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(Text.class);

    conf.setInputPath(new Path(points));
    Path outPath = new Path(output + "/clusters");
    conf.setOutputPath(outPath);

    conf.setMapperClass(ClusterMapper.class);
    conf.setReducerClass(IdentityReducer.class);

    client.setConf(conf);
    try {
      FileSystem dfs = FileSystem.get(conf);
      if (dfs.exists(outPath))
        dfs.delete(outPath);
      JobClient.runJob(conf);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

}
