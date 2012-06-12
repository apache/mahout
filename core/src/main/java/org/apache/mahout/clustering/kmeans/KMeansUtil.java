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

import java.util.Collection;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.canopy.Canopy;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

final class KMeansUtil {
  
  private static final Logger log = LoggerFactory.getLogger(KMeansUtil.class);

  private KMeansUtil() {}
  
  /**
   * Create a list of Klusters from whatever Cluster type is passed in as the prior
   * 
   * @param conf
   *          the Configuration
   * @param clusterPath
   *          the path to the prior Clusters
   * @param clusters
   *          a List<Cluster> to put values into
   */
  public static void configureWithClusterInfo(Configuration conf, Path clusterPath, Collection<Cluster> clusters) {
    for (Writable value : new SequenceFileDirValueIterable<Writable>(clusterPath, PathType.LIST,
        PathFilters.partFilter(), conf)) {
      Class<? extends Writable> valueClass = value.getClass();
      if (valueClass.equals(ClusterWritable.class)) {
        ClusterWritable clusterWritable = (ClusterWritable) value;
        value = clusterWritable.getValue();
        valueClass = value.getClass();
      }
      log.debug("Read 1 Cluster from {}", clusterPath);
      
      if (valueClass.equals(Kluster.class)) {
        // get the cluster info
        clusters.add((Kluster) value);
      } else if (valueClass.equals(Canopy.class)) {
        // get the cluster info
        Canopy canopy = (Canopy) value;
        clusters.add(new Kluster(canopy.getCenter(), canopy.getId(), canopy.getMeasure()));
      } else {
        throw new IllegalStateException("Bad value class: " + valueClass);
      }
    }
  }
  
}
