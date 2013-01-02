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

package org.apache.mahout.clustering.evaluation;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.classify.WeightedVectorWritable;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.math.VectorWritable;

public class RepresentativePointsMapper
  extends Mapper<IntWritable, WeightedVectorWritable, IntWritable, WeightedVectorWritable> {

  private Map<Integer, List<VectorWritable>> representativePoints;
  private final Map<Integer, WeightedVectorWritable> mostDistantPoints = Maps.newHashMap();
  private DistanceMeasure measure = new EuclideanDistanceMeasure();

  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    for (Map.Entry<Integer, WeightedVectorWritable> entry : mostDistantPoints.entrySet()) {
      context.write(new IntWritable(entry.getKey()), entry.getValue());
    }
    super.cleanup(context);
  }

  @Override
  protected void map(IntWritable clusterId, WeightedVectorWritable point, Context context)
    throws IOException, InterruptedException {
    mapPoint(clusterId, point, measure, representativePoints, mostDistantPoints);
  }

  public static void mapPoint(IntWritable clusterId,
                              WeightedVectorWritable point,
                              DistanceMeasure measure,
                              Map<Integer, List<VectorWritable>> representativePoints,
                              Map<Integer, WeightedVectorWritable> mostDistantPoints) {
    int key = clusterId.get();
    WeightedVectorWritable currentMDP = mostDistantPoints.get(key);

    List<VectorWritable> repPoints = representativePoints.get(key);
    double totalDistance = 0.0;
    if (repPoints != null) {
      for (VectorWritable refPoint : repPoints) {
        totalDistance += measure.distance(refPoint.get(), point.getVector());
      }
    }
    if (currentMDP == null || currentMDP.getWeight() < totalDistance) {
      mostDistantPoints.put(key, new WeightedVectorWritable(totalDistance, point.getVector().clone()));
    }
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    measure =
        ClassUtils.instantiateAs(conf.get(RepresentativePointsDriver.DISTANCE_MEASURE_KEY), DistanceMeasure.class);
    representativePoints = getRepresentativePoints(conf);
  }

  public void configure(Map<Integer, List<VectorWritable>> referencePoints, DistanceMeasure measure) {
    this.representativePoints = referencePoints;
    this.measure = measure;
  }

  public static Map<Integer, List<VectorWritable>> getRepresentativePoints(Configuration conf) {
    String statePath = conf.get(RepresentativePointsDriver.STATE_IN_KEY);
    return getRepresentativePoints(conf, new Path(statePath));
  }

  public static Map<Integer, List<VectorWritable>> getRepresentativePoints(Configuration conf, Path statePath) {
    Map<Integer, List<VectorWritable>> representativePoints = Maps.newHashMap();
    for (Pair<IntWritable,VectorWritable> record
         : new SequenceFileDirIterable<IntWritable,VectorWritable>(statePath,
                                                                   PathType.LIST,
                                                                   PathFilters.logsCRCFilter(),
                                                                   conf)) {
      int keyValue = record.getFirst().get();
      List<VectorWritable> repPoints = representativePoints.get(keyValue);
      if (repPoints == null) {
        repPoints = Lists.newArrayList();
        representativePoints.put(keyValue, repPoints);
      }
      repPoints.add(record.getSecond());
    }
    return representativePoints;
  }
}
