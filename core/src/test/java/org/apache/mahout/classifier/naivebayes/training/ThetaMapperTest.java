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

public class ThetaMapperTest extends MahoutTestCase {

  @Test
  public void standard() throws Exception {

    Mapper.Context ctx = EasyMock.createMock(Mapper.Context.class);
    AbstractThetaTrainer trainer = EasyMock.createMock(AbstractThetaTrainer.class);

    Vector instance1 = new DenseVector(new double[] { 1, 2, 3 });
    Vector instance2 = new DenseVector(new double[] { 4, 5, 6 });

    Vector perLabelThetaNormalizer = new DenseVector(new double[] { 7, 8 });

    ThetaMapper thetaMapper = new ThetaMapper();
    setField(thetaMapper, "trainer", trainer);

    trainer.train(0, instance1);
    trainer.train(1, instance2);
    EasyMock.expect(trainer.retrievePerLabelThetaNormalizer()).andReturn(perLabelThetaNormalizer);
    ctx.write(new Text(TrainNaiveBayesJob.LABEL_THETA_NORMALIZER), new VectorWritable(perLabelThetaNormalizer));

    EasyMock.replay(ctx, trainer);

    thetaMapper.map(new IntWritable(0), new VectorWritable(instance1), ctx);
    thetaMapper.map(new IntWritable(1), new VectorWritable(instance2), ctx);
    thetaMapper.cleanup(ctx);

    EasyMock.verify(ctx, trainer);
  }


}
