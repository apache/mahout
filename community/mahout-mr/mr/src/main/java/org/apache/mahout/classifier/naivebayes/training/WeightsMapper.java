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

package org.apache.mahout.classifier.naivebayes.training;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

import com.google.common.base.Preconditions;

public class WeightsMapper extends Mapper<IntWritable, VectorWritable, Text, VectorWritable> {

  static final String NUM_LABELS = WeightsMapper.class.getName() + ".numLabels";

  private Vector weightsPerFeature;
  private Vector weightsPerLabel;

  @Override
  protected void setup(Context ctx) throws IOException, InterruptedException {
    super.setup(ctx);
    int numLabels = Integer.parseInt(ctx.getConfiguration().get(NUM_LABELS));
    Preconditions.checkArgument(numLabels > 0, "Wrong numLabels: " + numLabels + ". Must be > 0!");
    weightsPerLabel = new DenseVector(numLabels);
  }

  @Override
  protected void map(IntWritable index, VectorWritable value, Context ctx) throws IOException, InterruptedException {
    Vector instance = value.get();
    if (weightsPerFeature == null) {
      weightsPerFeature = new RandomAccessSparseVector(instance.size(), instance.getNumNondefaultElements());
    }

    int label = index.get();
    weightsPerFeature.assign(instance, Functions.PLUS);
    weightsPerLabel.set(label, weightsPerLabel.get(label) + instance.zSum());
  }

  @Override
  protected void cleanup(Context ctx) throws IOException, InterruptedException {
    if (weightsPerFeature != null) {
      ctx.write(new Text(TrainNaiveBayesJob.WEIGHTS_PER_FEATURE), new VectorWritable(weightsPerFeature));
      ctx.write(new Text(TrainNaiveBayesJob.WEIGHTS_PER_LABEL), new VectorWritable(weightsPerLabel));
    }
    super.cleanup(ctx);
  }
}
