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

package org.apache.mahout.clustering.meanshift;

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

public class MeanShiftCanopyClusterMapper
  extends Mapper<WritableComparable<?>, MeanShiftCanopy, IntWritable, WeightedVectorWritable> {

  private List<MeanShiftCanopy> canopies;

  @Override
  protected void map(WritableComparable<?> key, MeanShiftCanopy canopy, Context context)
    throws IOException, InterruptedException {
    // canopies use canopyIds assigned when input vectors are processed as vectorIds too
    int vectorId = canopy.getId();
    for (MeanShiftCanopy msc : canopies) {
      for (int containedId : msc.getBoundPoints().toList()) {
        if (vectorId == containedId) {
          context.write(new IntWritable(msc.getId()),
                         new WeightedVectorWritable(1, canopy.getCenter()));
        }
      }
    }
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    canopies = getCanopies(context.getConfiguration());
  }

  public static List<MeanShiftCanopy> getCanopies(Configuration conf) {
    String statePath = conf.get(MeanShiftCanopyDriver.STATE_IN_KEY);
    List<MeanShiftCanopy> canopies = Lists.newArrayList();
    Path path = new Path(statePath);
    for (MeanShiftCanopy value 
         : new SequenceFileDirValueIterable<MeanShiftCanopy>(path, PathType.LIST, PathFilters.logsCRCFilter(), conf)) {
      canopies.add(value);
    }
    return canopies;
  }

}
