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

package org.apache.mahout.clustering.canopy;

import java.io.IOException;
import java.util.Collection;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class ClusterMapper
    extends Mapper<WritableComparable<?>, VectorWritable, IntWritable, WeightedVectorWritable> {

  private CanopyClusterer canopyClusterer;

  @Override
  protected void map(WritableComparable<?> key, VectorWritable point,
      Context context) throws IOException, InterruptedException {
    canopyClusterer.emitPointToClosestCanopy(point.get(), canopies, context);
  }

  private final Collection<Canopy> canopies = Lists.newArrayList();

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);

    canopyClusterer = new CanopyClusterer(context.getConfiguration());

    Configuration conf = context.getConfiguration();
    String clustersIn = conf.get(CanopyConfigKeys.CANOPY_PATH_KEY);

    // filter out the files
    if (clustersIn != null && clustersIn.length() > 0) {
      Path clusterPath = new Path(clustersIn, "*");
      FileSystem fs = clusterPath.getFileSystem(conf);
      Path[] paths = FileUtil.stat2Paths(fs.globStatus(clusterPath, PathFilters.partFilter()));
      for (FileStatus file : fs.listStatus(paths, PathFilters.partFilter())) {
        for (Canopy value : new SequenceFileValueIterable<Canopy>(file.getPath(), conf)) {
          canopies.add(value);
        }
      }

      if (canopies.isEmpty()) {
        throw new IllegalStateException("Canopies are empty!");
      }
    }

  }

  public boolean canopyCovers(Canopy canopy, Vector point) {
    return canopyClusterer.canopyCovers(canopy, point);
  }
}
