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

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.easymock.EasyMock;
import org.junit.Test;

public class WeightsMapperTest extends MahoutTestCase {

  @Test
  public void scores() throws Exception {

    Mapper.Context ctx = EasyMock.createMock(Mapper.Context.class);
    Vector instance1 = new DenseVector(new double[] { 1, 0,   0.5, 0.5, 0 });
    Vector instance2 = new DenseVector(new double[] { 0, 0.5, 0,   0,   0 });
    Vector instance3 = new DenseVector(new double[] { 1, 0.5, 1,   1.5, 1 });

    Vector weightsPerLabel = new DenseVector(new double[] { 0, 0 });

    ctx.write(new Text(TrainNaiveBayesJob.WEIGHTS_PER_FEATURE),
        new VectorWritable(new DenseVector(new double[] { 2, 1, 1.5, 2, 1 })));
    ctx.write(new Text(TrainNaiveBayesJob.WEIGHTS_PER_LABEL),
        new VectorWritable(new DenseVector(new double[] { 2.5, 5 })));

    EasyMock.replay(ctx);

    WeightsMapper weights = new WeightsMapper();
    setField(weights, "weightsPerLabel", weightsPerLabel);

    weights.map(new IntWritable(0), new VectorWritable(instance1), ctx);
    weights.map(new IntWritable(0), new VectorWritable(instance2), ctx);
    weights.map(new IntWritable(1), new VectorWritable(instance3), ctx);

    weights.cleanup(ctx);

    EasyMock.verify(ctx);
  }
}
