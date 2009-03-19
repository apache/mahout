package org.apache.mahout.clustering.kmeans;
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

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class KMeansUtil {
  private static final Logger log = LoggerFactory.getLogger(KMeansUtil.class);

  /**
   * Configure the mapper with the cluster info
   * 
   * @param job
   * @param clusters
   */
  public static void configureWithClusterInfo(String clusterPathStr,
      List<Cluster> clusters) {    
    
    // Get the path location where the cluster Info is stored
    JobConf job = new JobConf(KMeansUtil.class);
    Path clusterPath = new Path(clusterPathStr);
    List<Path> result = new ArrayList<Path>();

    // filter out the files
    PathFilter clusterFileFilter = new PathFilter() {
      public boolean accept(Path path) {
        return path.getName().startsWith("part");
      }
    };

    try {
      // get all filtered file names in result list
      FileSystem fs = clusterPath.getFileSystem(job);
      FileStatus[] matches = fs.listStatus(FileUtil.stat2Paths(fs.globStatus(
          clusterPath, clusterFileFilter)), clusterFileFilter);

      for (FileStatus match : matches) {
        result.add(fs.makeQualified(match.getPath()));
      }

      // iterate thru the result path list
      for (Path path : result) {
        SequenceFile.Reader reader = null;
//        RecordReader<Text, Text> recordReader = null;
        try {
          reader =new SequenceFile.Reader(fs, path, job); 
          Text key = new Text();
          Text value = new Text();
          int counter = 1;
          while (reader.next(key, value)) {
            // get the cluster info
            Cluster cluster = Cluster.decodeCluster(value.toString());
            clusters.add(cluster);
          }
        } finally {
          if (reader != null) {
            reader.close();
          }
        }
      }

    } catch (IOException e) {
      log.info("Exception occurred in loading clusters:", e);
      e.printStackTrace();
      throw new RuntimeException(e);
    }
  }

}
