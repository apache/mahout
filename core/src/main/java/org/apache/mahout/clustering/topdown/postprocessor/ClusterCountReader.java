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

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterator;

/**
 * Reads the number of clusters produced by the clustering algorithm.
 */
public final class ClusterCountReader {

  private ClusterCountReader() {
  }

  /**
   * Reads the number of clusters present by reading the clusters-*-final file.
   * 
   * @param clusterOutputPath
   *          The output path provided to the clustering algorithm.
   * @param conf
   *          The hadoop configuration.
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

}
