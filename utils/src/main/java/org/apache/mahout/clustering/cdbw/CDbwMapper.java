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

package org.apache.mahout.clustering.cdbw;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.OutputLogFilter;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.VectorWritable;

public class CDbwMapper extends Mapper<IntWritable, WeightedVectorWritable, IntWritable, WeightedVectorWritable> {

  private Map<Integer, List<VectorWritable>> representativePoints;

  private final Map<Integer, WeightedVectorWritable> mostDistantPoints = new HashMap<Integer, WeightedVectorWritable>();

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
    int key = clusterId.get();
    WeightedVectorWritable currentMDP = mostDistantPoints.get(key);

    List<VectorWritable> refPoints = representativePoints.get(key);
    if (refPoints == null){
      System.out.println();
    }
    double totalDistance = 0.0;
    for (VectorWritable refPoint : refPoints) {
      totalDistance += measure.distance(refPoint.get(), point.getVector());
    }
    if (currentMDP == null || currentMDP.getWeight() < totalDistance) {
      mostDistantPoints.put(key, new WeightedVectorWritable(totalDistance, point.getVector().clone()));
    }
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    try {
      ClassLoader ccl = Thread.currentThread().getContextClassLoader();
      measure = ccl.loadClass(conf.get(CDbwDriver.DISTANCE_MEASURE_KEY))
          .asSubclass(DistanceMeasure.class).newInstance();
      representativePoints = getRepresentativePoints(conf);
    } catch (NumberFormatException e) {
      throw new IllegalStateException(e);
    } catch (SecurityException e) {
      throw new IllegalStateException(e);
    } catch (IllegalArgumentException e) {
      throw new IllegalStateException(e);
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    }
  }

  public void configure(Map<Integer, List<VectorWritable>> referencePoints, DistanceMeasure measure) {
    this.representativePoints = referencePoints;
    this.measure = measure;
  }

  public static Map<Integer, List<VectorWritable>> getRepresentativePoints(Configuration conf) {
    String statePath = conf.get(CDbwDriver.STATE_IN_KEY);
    Map<Integer, List<VectorWritable>> representativePoints = new HashMap<Integer, List<VectorWritable>>();
    try {
      Path path = new Path(statePath);
      FileSystem fs = FileSystem.get(path.toUri(), conf);
      FileStatus[] status = fs.listStatus(path, new OutputLogFilter());
      for (FileStatus s : status) {
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, s.getPath(), conf);
        try {
          IntWritable key = new IntWritable(0);
          VectorWritable point = new VectorWritable();
          while (reader.next(key, point)) {
            List<VectorWritable> repPoints = representativePoints.get(key.get());
            if (repPoints == null) {
              repPoints = new ArrayList<VectorWritable>();
              representativePoints.put(key.get(), repPoints);
            }
            repPoints.add(point);
            point = new VectorWritable();
          }
        } finally {
          reader.close();
        }
      }
      return representativePoints;
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }
}
