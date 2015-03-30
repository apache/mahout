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

import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.clustering.classify.WeightedVectorWritable;
import org.apache.mahout.clustering.topdown.PathDirectory;
import org.apache.mahout.common.IOUtils;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.Map;

/**
 * This class reads the output of any clustering algorithm, and, creates separate directories for different
 * clusters. Each cluster directory's name is its clusterId. Each and every point is written in the cluster
 * directory associated with that point.
 * <p/>
 * This class incorporates a sequential algorithm and is appropriate for use for data which has been clustered
 * sequentially.
 * <p/>
 * The sequential and non sequential version, both are being used from {@link ClusterOutputPostProcessorDriver}.
 */
public final class ClusterOutputPostProcessor {

  private Path clusteredPoints;
  private final FileSystem fileSystem;
  private final Configuration conf;
  private final Path clusterPostProcessorOutput;
  private final Map<String, Path> postProcessedClusterDirectories = Maps.newHashMap();
  private long uniqueVectorId = 0L;
  private final Map<String, SequenceFile.Writer> writersForClusters;

  public ClusterOutputPostProcessor(Path clusterOutputToBeProcessed,
                                    Path output,
                                    Configuration hadoopConfiguration) throws IOException {
    this.clusterPostProcessorOutput = output;
    this.clusteredPoints = PathDirectory.getClusterOutputClusteredPoints(clusterOutputToBeProcessed);
    this.conf = hadoopConfiguration;
    this.writersForClusters = Maps.newHashMap();
    fileSystem = clusteredPoints.getFileSystem(conf);
  }

  /**
   * This method takes the clustered points output by the clustering algorithms as input and writes them into
   * their respective clusters.
   */
  public void process() throws IOException {
    createPostProcessDirectory();
    for (Pair<?, WeightedVectorWritable> record
        : new SequenceFileDirIterable<Writable, WeightedVectorWritable>(clusteredPoints, PathType.GLOB, PathFilters.partFilter(),
                                                                        null, false, conf)) {
      String clusterId = record.getFirst().toString().trim();
      putVectorInRespectiveCluster(clusterId, record.getSecond());
    }
    IOUtils.close(writersForClusters.values());
    writersForClusters.clear();
  }

  /**
   * Creates the directory to put post processed clusters.
   */
  private void createPostProcessDirectory() throws IOException {
    if (!fileSystem.exists(clusterPostProcessorOutput)
            && !fileSystem.mkdirs(clusterPostProcessorOutput)) {
      throw new IOException("Error creating cluster post processor directory");
    }
  }

  /**
   * Finds out the cluster directory of the vector and writes it into the specified cluster.
   */
  private void putVectorInRespectiveCluster(String clusterId, WeightedVectorWritable point) throws IOException {
    Writer writer = findWriterForVector(clusterId);
    postProcessedClusterDirectories.put(clusterId,
            PathDirectory.getClusterPathForClusterId(clusterPostProcessorOutput, clusterId));
    writeVectorToCluster(writer, point);
  }

  /**
   * Finds out the path in cluster where the point is supposed to be written.
   */
  private Writer findWriterForVector(String clusterId) throws IOException {
    Path clusterDirectory = PathDirectory.getClusterPathForClusterId(clusterPostProcessorOutput, clusterId);
    Writer writer = writersForClusters.get(clusterId);
    if (writer == null) {
      Path pathToWrite = new Path(clusterDirectory, new Path("part-m-0"));
      writer = new Writer(fileSystem, conf, pathToWrite, LongWritable.class, VectorWritable.class);
      writersForClusters.put(clusterId, writer);
    }
    return writer;
  }

  /**
   * Writes vector to the cluster directory.
   */
  private void writeVectorToCluster(Writer writer, WeightedVectorWritable point) throws IOException {
    writer.append(new LongWritable(uniqueVectorId++), new VectorWritable(point.getVector()));
    writer.sync();
  }

  /**
   * @return the set of all post processed cluster paths.
   */
  public Map<String, Path> getPostProcessedClusterDirectories() {
    return postProcessedClusterDirectories;
  }

  public void setClusteredPoints(Path clusteredPoints) {
    this.clusteredPoints = clusteredPoints;
  }

}
