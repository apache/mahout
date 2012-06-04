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
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.map.OpenObjectIntHashMap;
import org.easymock.EasyMock;
import org.junit.Before;
import org.junit.Test;

public class IndexInstancesMapperTest extends MahoutTestCase {

  private Mapper.Context ctx;
  private OpenObjectIntHashMap<String> labelIndex;
  private VectorWritable instance;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();

    ctx = EasyMock.createMock(Mapper.Context.class);
    instance = new VectorWritable(new DenseVector(new double[] { 1, 0, 1, 1, 0 }));

    labelIndex = new OpenObjectIntHashMap<String>();
    labelIndex.put("bird", 0);
    labelIndex.put("cat", 1);
  }


  @Test
  public void index() throws Exception {

    ctx.write(new IntWritable(0), instance);

    EasyMock.replay(ctx);

    IndexInstancesMapper indexInstances = new IndexInstancesMapper();
    setField(indexInstances, "labelIndex", labelIndex);

    indexInstances.map(new Text("/bird/"), instance, ctx);

    EasyMock.verify(ctx);
  }

  @Test
  public void skip() throws Exception {

    Counter skippedInstances = EasyMock.createMock(Counter.class);

    EasyMock.expect(ctx.getCounter(IndexInstancesMapper.Counter.SKIPPED_INSTANCES)).andReturn(skippedInstances);
    skippedInstances.increment(1);

    EasyMock.replay(ctx, skippedInstances);

    IndexInstancesMapper indexInstances = new IndexInstancesMapper();
    setField(indexInstances, "labelIndex", labelIndex);

    indexInstances.map(new Text("/fish/"), instance, ctx);

    EasyMock.verify(ctx, skippedInstances);
  }

}
