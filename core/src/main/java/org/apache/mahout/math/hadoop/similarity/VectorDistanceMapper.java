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

package org.apache.mahout.math.hadoop.similarity;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.List;

public final class VectorDistanceMapper
    extends Mapper<WritableComparable<?>, VectorWritable, StringTuple, DoubleWritable> {

  private DistanceMeasure measure;
  private List<NamedVector> seedVectors;
  private boolean usesThreshold = false;
  private double maxDistance;

  @Override
  protected void map(WritableComparable<?> key, VectorWritable value, Context context)
    throws IOException, InterruptedException {
    String keyName;
    Vector valVec = value.get();
    if (valVec instanceof NamedVector) {
      keyName = ((NamedVector) valVec).getName();
    } else {
      keyName = key.toString();
    }
    
    for (NamedVector seedVector : seedVectors) {
      double distance = measure.distance(seedVector, valVec);
      if (!usesThreshold || distance <= maxDistance) {
        StringTuple outKey = new StringTuple();
        outKey.add(seedVector.getName());
        outKey.add(keyName);
        context.write(outKey, new DoubleWritable(distance));
      }
    }
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();

    String maxDistanceParam = conf.get(VectorDistanceSimilarityJob.MAX_DISTANCE);
    if (maxDistanceParam != null) {
      usesThreshold = true;
      maxDistance = Double.parseDouble(maxDistanceParam);
    }
    
    measure = ClassUtils.instantiateAs(conf.get(VectorDistanceSimilarityJob.DISTANCE_MEASURE_KEY),
        DistanceMeasure.class);
    measure.configure(conf);
    seedVectors = SeedVectorUtil.loadSeedVectors(conf);
  }
}
