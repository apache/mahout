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
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class ClusterMapper extends Mapper<WritableComparable<?>, VectorWritable, IntWritable, WeightedVectorWritable> {

  private CanopyClusterer canopyClusterer;

  @Override
  protected void map(WritableComparable<?> key, VectorWritable point, Context context) throws IOException, InterruptedException {
    canopyClusterer.emitPointToClosestCanopy(point.get(), canopies, context);
  }

  private final List<Canopy> canopies = new ArrayList<Canopy>();

  /**
   * Configure the mapper by providing its canopies. Used by unit tests.
   * 
   * @param canopies
   *          a List<Canopy>
   */
  public void config(Collection<Canopy> canopies) {
    this.canopies.clear();
    this.canopies.addAll(canopies);
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);

    canopyClusterer = new CanopyClusterer(context.getConfiguration());

    Configuration configuration = context.getConfiguration();
    String canopyPath = configuration.get(CanopyConfigKeys.CANOPY_PATH_KEY);

    if ((canopyPath != null) && (canopyPath.length() > 0)) {
      try {
        Path path = new Path(canopyPath);
        FileSystem fs = FileSystem.get(path.toUri(), configuration);
        FileStatus[] files = fs.listStatus(path);
        for (FileStatus file : files) {
          SequenceFile.Reader reader = new SequenceFile.Reader(fs, file.getPath(), configuration);
          try {
            Text key = new Text();
            Canopy value = new Canopy();
            while (reader.next(key, value)) {
              canopies.add(value);
              value = new Canopy();
            }
          } finally {
            reader.close();
          }
        }
      } catch (IOException e) {
        throw new IllegalStateException(e);
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
