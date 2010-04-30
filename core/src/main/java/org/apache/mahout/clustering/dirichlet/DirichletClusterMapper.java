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

package org.apache.mahout.clustering.dirichlet;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.OutputLogFilter;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class DirichletClusterMapper extends MapReduceBase implements
    Mapper<WritableComparable<?>, VectorWritable, IntWritable, WeightedVectorWritable> {

  private OutputCollector<IntWritable, VectorWritable> output;

  private List<DirichletCluster> clusters;

  private DirichletClusterer clusterer;

  @SuppressWarnings("unchecked")
  @Override
  public void map(WritableComparable<?> key, VectorWritable vector, OutputCollector<IntWritable, WeightedVectorWritable> output,
      Reporter reporter) throws IOException {
    clusterer.emitPointToClusters(vector, clusters, output);
  }

  @Override
  public void configure(JobConf job) {
    super.configure(job);
    try {
      clusters = getClusters(job);
      String emitMostLikely = job.get(DirichletDriver.EMIT_MOST_LIKELY_KEY);
      String threshold = job.get(DirichletDriver.THRESHOLD_KEY);
      clusterer = new DirichletClusterer<Vector>(Boolean.parseBoolean(emitMostLikely), Double.parseDouble(threshold));
    } catch (SecurityException e) {
      throw new IllegalStateException(e);
    } catch (IllegalArgumentException e) {
      throw new IllegalStateException(e);
    } catch (NoSuchMethodException e) {
      throw new IllegalStateException(e);
    } catch (InvocationTargetException e) {
      throw new IllegalStateException(e);
    }
  }

  public static List<DirichletCluster> getClusters(JobConf job) throws SecurityException, IllegalArgumentException,
      NoSuchMethodException, InvocationTargetException {
    String statePath = job.get(DirichletDriver.STATE_IN_KEY);
    List<DirichletCluster> clusters = new ArrayList<DirichletCluster>();
    try {
      Path path = new Path(statePath);
      FileSystem fs = FileSystem.get(path.toUri(), job);
      FileStatus[] status = fs.listStatus(path, new OutputLogFilter());
      for (FileStatus s : status) {
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, s.getPath(), job);
        try {
          Text key = new Text();
          DirichletCluster cluster = new DirichletCluster();
          while (reader.next(key, cluster)) {
            clusters.add(cluster);
            cluster = new DirichletCluster();
          }
        } finally {
          reader.close();
        }
      }
      return clusters;
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

}
