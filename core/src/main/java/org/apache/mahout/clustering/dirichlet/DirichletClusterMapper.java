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
import java.util.List;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.math.VectorWritable;

public class DirichletClusterMapper
  extends Mapper<WritableComparable<?>, VectorWritable, IntWritable, WeightedVectorWritable> {

  private List<DirichletCluster> clusters;

  private DirichletClusterer clusterer;

  @Override
  protected void map(WritableComparable<?> key, VectorWritable vector, Context context)
    throws IOException, InterruptedException {
    clusterer.emitPointToClusters(vector, clusters, context);
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    clusters = getClusters(conf);
    for (DirichletCluster cluster : clusters) {
      cluster.getModel().configure(conf);
    }
    String emitMostLikely = conf.get(DirichletDriver.EMIT_MOST_LIKELY_KEY);
    String threshold = conf.get(DirichletDriver.THRESHOLD_KEY);
    clusterer = new DirichletClusterer(Boolean.parseBoolean(emitMostLikely), Double.parseDouble(threshold));
  }

  public static List<DirichletCluster> getClusters(Configuration conf) {
    String statePath = conf.get(DirichletDriver.STATE_IN_KEY);
    return loadClusters(conf, new Path(statePath));
  }

  protected static List<DirichletCluster> loadClusters(Configuration conf, Path stateIn) {
    List<DirichletCluster> clusters = Lists.newArrayList();
    for (DirichletCluster value
         : new SequenceFileDirValueIterable<DirichletCluster>(stateIn,
                                                              PathType.LIST,
                                                              PathFilters.logsCRCFilter(),
                                                              conf)) {
      clusters.add(value);
    }
    return clusters;
  }

}
