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
package org.apache.mahout.clustering.fuzzykmeans;

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
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.matrix.AbstractVector;
import org.apache.mahout.matrix.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FuzzyKMeansMapper extends MapReduceBase implements
    Mapper<WritableComparable, Text, Text, Text> {

  private static final Logger log = LoggerFactory
      .getLogger(FuzzyKMeansMapper.class);

  protected List<SoftCluster> clusters;

  public void map(WritableComparable key, Text values,
      OutputCollector<Text, Text> output, Reporter reporter) throws IOException {
    Vector point = AbstractVector.decodeVector(values.toString());
    SoftCluster.emitPointProbToCluster(point, clusters, values, output);
  }

  /**
   * Configure the mapper by providing its clusters. Used by unit tests.
   * 
   * @param clusters a List<Cluster>
   */
  void config(List<SoftCluster> clusters) {
    this.clusters = clusters;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.hadoop.mapred.MapReduceBase#configure(org.apache.hadoop.mapred.JobConf)
   */
  @Override
  public void configure(JobConf job) {

    super.configure(job);
    SoftCluster.configure(job);

    log.info("In Mapper Configure:");
    clusters = new ArrayList<SoftCluster>();

    configureWithClusterInfo(job);

    if (clusters.size() == 0)
      throw new NullPointerException("Cluster is empty!!!");
  }

  /**
   * Configure the mapper with the cluster info
   * 
   * @param job
   */
  protected void configureWithClusterInfo(JobConf job) {
    // Get the path location where the cluster Info is stored
    String clusterPathStr = job.get(SoftCluster.CLUSTER_PATH_KEY);
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
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, job);
        try {
          Text key = new Text();
          Text value = new Text();
          //int counter = 1;
          while (reader.next(key, value)) {
            // get the cluster info
            SoftCluster cluster = SoftCluster.decodeCluster(value.toString());
            // add the center so the centroid will be correct on output
            // formatting
            cluster.addPoint(cluster.getCenter(), 1);
            clusters.add(cluster);
          }
        } finally {
          reader.close();
        }
      }

    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
