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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;

/**
 * Reads the number of clusters produced by the clustering algorithm.
 */
public class ClusterCountReader {
  
  /**
   * Reads the number of clusters present by reading the clusters-*-final file.
   * 
   * @param clusterOutputPath
   *          The output path provided to the clustering algorithm.
   * @param conf
   *          The hadoop configuration.
   * @return the number of final clusters.
   * @throws IOException
   * @throws IllegalAccessException
   * @throws InstantiationException
   */
  public static int getNumberOfClusters(Path clusterOutputPath, Configuration conf) throws IOException,
                                                                                   InstantiationException,
                                                                                   IllegalAccessException {
    int numberOfClusters = 0;
    FileStatus[] partFiles = getPartFiles(clusterOutputPath, conf);
    for (FileStatus fileStatus : partFiles) {
      SequenceFile.Reader reader = new SequenceFile.Reader(FileSystem.get(conf), fileStatus.getPath(), conf);
      WritableComparable key = (WritableComparable) reader.getKeyClass().newInstance();
      Writable value = (Writable) reader.getValueClass().newInstance();
      while (reader.next(key, value)) {
        numberOfClusters++;
      }
      reader.close();
    }
    return numberOfClusters;
  }
  
  /**
   * Gets the part file of the final iteration. clusters-n-final
   * 
   */
  private static FileStatus[] getPartFiles(Path path, Configuration conf) throws IOException {
    FileSystem fileSystem = path.getFileSystem(conf);
    FileStatus[] clusterFiles = fileSystem.listStatus(path, CLUSTER_FINAL);
    FileStatus[] partFileStatuses = fileSystem
        .listStatus(clusterFiles[0].getPath(), PathFilters.partFilter());
    return partFileStatuses;
  }
  
  /**
   * Pathfilter to read the final clustering file.
   */
  private static final PathFilter CLUSTER_FINAL = new PathFilter() {
    @Override
    public boolean accept(Path path) {
      String name = path.getName();
      return name.startsWith("clusters-") && name.endsWith("-final");
    }
  };
}