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

package org.apache.mahout.clustering.fuzzykmeans;

import java.util.Collection;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.clustering.canopy.Canopy;
import org.apache.mahout.clustering.kmeans.Cluster;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;

final class FuzzyKMeansUtil {

  private FuzzyKMeansUtil() {
  }

  /** Configure the mapper with the cluster info */
  public static void configureWithClusterInfo(Path clusterPath, Collection<SoftCluster> clusters) {
    for (Writable value :
         new SequenceFileDirValueIterable<Writable>(clusterPath,
                                                    PathType.LIST,
                                                    PathFilters.partFilter(),
                                                    new Configuration())) {
      Class<? extends Writable> valueClass = value.getClass();
      if (valueClass.equals(Cluster.class)) {
        // get the cluster info
        Cluster cluster = (Cluster) value;
        clusters.add(new SoftCluster(cluster.getCenter(), cluster.getId(), cluster.getMeasure()));
      } else if (valueClass.equals(SoftCluster.class)) {
        // get the cluster info
        clusters.add((SoftCluster) value);
      } else if (valueClass.equals(Canopy.class)) {
        // get the cluster info
        Canopy canopy = (Canopy) value;
        clusters.add(new SoftCluster(canopy.getCenter(), canopy.getId(), canopy.getMeasure()));
      } else {
        throw new IllegalStateException("Bad value class: " + valueClass);
      }
    }

  }

}
