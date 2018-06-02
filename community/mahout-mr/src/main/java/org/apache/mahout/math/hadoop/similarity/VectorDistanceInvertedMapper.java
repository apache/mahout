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
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.List;

/**
 * Similar to {@link VectorDistanceMapper}, except it outputs
 * &lt;input, Vector&gt;, where the vector is a dense vector contain one entry for every seed vector
 */
public final class VectorDistanceInvertedMapper
    extends Mapper<WritableComparable<?>, VectorWritable, Text, VectorWritable> {

  private DistanceMeasure measure;
  private List<NamedVector> seedVectors;

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
    Vector outVec = new DenseVector(new double[seedVectors.size()]);
    int i = 0;
    for (NamedVector seedVector : seedVectors) {
      outVec.setQuick(i++, measure.distance(seedVector, valVec));
    }
    context.write(new Text(keyName), new VectorWritable(outVec));
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    measure =
        ClassUtils.instantiateAs(conf.get(VectorDistanceSimilarityJob.DISTANCE_MEASURE_KEY), DistanceMeasure.class);
    measure.configure(conf);
    seedVectors = SeedVectorUtil.loadSeedVectors(conf);
  }
}
