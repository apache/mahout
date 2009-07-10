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

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.clustering.canopy.Canopy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

final class KMeansUtil {

  private static final Logger log = LoggerFactory.getLogger(KMeansUtil.class);

  private KMeansUtil() {
  }

  /** Configure the mapper with the cluster info */
  public static void configureWithClusterInfo(String clusterPathStr,
                                              List<Cluster> clusters) {

    // Get the path location where the cluster Info is stored
    JobConf job = new JobConf(KMeansUtil.class);
    Path clusterPath = new Path(clusterPathStr + "/*");
    List<Path> result = new ArrayList<Path>();

    // filter out the files
    PathFilter clusterFileFilter = new PathFilter() {
      @Override
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
        try {
          reader = new SequenceFile.Reader(fs, path, job);
          Class<?> valueClass = reader.getValueClass();
          Writable key;
          try {
            key = (Writable) reader.getKeyClass().newInstance();
          } catch (InstantiationException e) {//Should not be possible
            log.error("Exception", e);
            throw new RuntimeException(e);
          } catch (IllegalAccessException e) {
            log.error("Exception", e);
            throw new RuntimeException(e);
          }
          if (valueClass.equals(Cluster.class)) {
            Cluster value = new Cluster();
            while (reader.next(key, value)) {
              // get the cluster info
              clusters.add(value);
              value = new Cluster();
            }
          } else if (valueClass.equals(Canopy.class)) {
            Canopy value = new Canopy();
            while (reader.next(key, value)) {
              // get the cluster info
              Cluster cluster = new Cluster(value.getCenter(), value.getId());
              clusters.add(cluster);
              value = new Canopy();
            }
          }
        } finally {
          if (reader != null) {
            reader.close();
          }
        }
      }

    } catch (IOException e) {
      log.info("Exception occurred in loading clusters:", e);
      throw new RuntimeException(e);
    }
  }

}
