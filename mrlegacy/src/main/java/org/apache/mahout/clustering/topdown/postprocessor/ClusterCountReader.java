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

package org.apache.mahout.clustering.topdown.postprocessor;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterator;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 * Reads the number of clusters produced by the clustering algorithm.
 */
public final class ClusterCountReader {

  private ClusterCountReader() {
  }

  /**
   * Reads the number of clusters present by reading the clusters-*-final file.
   *
   * @param clusterOutputPath The output path provided to the clustering algorithm.
   * @param conf              The hadoop configuration.
   * @return the number of final clusters.
   */
  public static int getNumberOfClusters(Path clusterOutputPath, Configuration conf) throws IOException {
    FileSystem fileSystem = clusterOutputPath.getFileSystem(conf);
    FileStatus[] clusterFiles = fileSystem.listStatus(clusterOutputPath, PathFilters.finalPartFilter());
    int numberOfClusters = 0;
    Iterator<?> it = new SequenceFileDirValueIterator<Writable>(clusterFiles[0].getPath(),
            PathType.LIST,
            PathFilters.partFilter(),
            null,
            true,
            conf);
    while (it.hasNext()) {
      it.next();
      numberOfClusters++;
    }
    return numberOfClusters;
  }

  /**
   * Generates a list of all cluster ids by reading the clusters-*-final file.
   *
   * @param clusterOutputPath The output path provided to the clustering algorithm.
   * @param conf              The hadoop configuration.
   * @return An ArrayList containing the final cluster ids.
   */
  public static Map<Integer, Integer> getClusterIDs(Path clusterOutputPath, Configuration conf, boolean keyIsClusterId)
    throws IOException {
    Map<Integer, Integer> clusterIds = new HashMap<Integer, Integer>();
    FileSystem fileSystem = clusterOutputPath.getFileSystem(conf);
    FileStatus[] clusterFiles = fileSystem.listStatus(clusterOutputPath, PathFilters.finalPartFilter());
    //System.out.println("LOOK HERE: " + clusterOutputPath);
    Iterator<ClusterWritable> it = new SequenceFileDirValueIterator<ClusterWritable>(clusterFiles[0].getPath(),
            PathType.LIST,
            PathFilters.partFilter(),
            null,
            true,
            conf);
    int i = 0;
    while (it.hasNext()) {
      Integer key;
      Integer value;
      if (keyIsClusterId) { // key is the cluster id, value is i, the index we will use
        key = it.next().getValue().getId();
        value = i;
      } else {
        key = i;
        value = it.next().getValue().getId();
      }
      clusterIds.put(key, value);
      i++;
    }
    return clusterIds;
  }

}
