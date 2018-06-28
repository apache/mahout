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

package org.apache.mahout.clustering.topdown;

import java.io.File;

import org.apache.hadoop.fs.Path;

/**
 * Contains list of all internal paths used in top down clustering.
 */
public final class PathDirectory {

  public static final String TOP_LEVEL_CLUSTER_DIRECTORY = "topLevelCluster";
  public static final String POST_PROCESS_DIRECTORY = "clusterPostProcessed";
  public static final String CLUSTERED_POINTS_DIRECTORY = "clusteredPoints";
  public static final String BOTTOM_LEVEL_CLUSTER_DIRECTORY = "bottomLevelCluster";

  private PathDirectory() {
  }

  /**
   * All output of top level clustering is stored in output directory/topLevelCluster.
   * 
   * @param output
   *          the output path of clustering.
   * @return The top level Cluster Directory.
   */
  public static Path getTopLevelClusterPath(Path output) {
    return new Path(output + File.separator + TOP_LEVEL_CLUSTER_DIRECTORY);
  }
  
  /**
   * The output of top level clusters is post processed and kept in this path.
   * 
   * @param outputPathProvidedByUser
   *          the output path of clustering.
   * @return the path where the output of top level cluster post processor is kept.
   */
  public static Path getClusterPostProcessorOutputDirectory(Path outputPathProvidedByUser) {
    return new Path(outputPathProvidedByUser + File.separator + POST_PROCESS_DIRECTORY);
  }
  
  /**
   * The top level clustered points before post processing is generated here.
   * 
   * @param output
   *          the output path of clustering.
   * @return the clustered points directory
   */
  public static Path getClusterOutputClusteredPoints(Path output) {
    return new Path(output + File.separator + CLUSTERED_POINTS_DIRECTORY + File.separator, "*");
  }
  
  /**
   * Each cluster produced by top level clustering is processed in output/"bottomLevelCluster"/clusterId.
   * 
   * @param output
   * @param clusterId
   * @return the bottom level clustering path.
   */
  public static Path getBottomLevelClusterPath(Path output, String clusterId) {
    return new Path(output + File.separator + BOTTOM_LEVEL_CLUSTER_DIRECTORY + File.separator + clusterId);
  }
  
  /**
   * Each clusters path name is its clusterId. The vectors reside in separate files inside it.
   * 
   * @param clusterPostProcessorOutput
   *          the path of cluster post processor output.
   * @param clusterId
   *          the id of the cluster.
   * @return the cluster path for cluster id.
   */
  public static Path getClusterPathForClusterId(Path clusterPostProcessorOutput, String clusterId) {
    return new Path(clusterPostProcessorOutput + File.separator + clusterId);
  }
  
}
