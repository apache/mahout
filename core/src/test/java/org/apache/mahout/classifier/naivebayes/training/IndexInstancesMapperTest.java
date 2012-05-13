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
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.MultiLabelVectorWritable;
import org.apache.mahout.math.VectorWritable;
import org.easymock.EasyMock;
import org.junit.Before;
import org.junit.Test;

@SuppressWarnings("unchecked")
public class IndexInstancesMapperTest extends MahoutTestCase {
  private static final DenseVector VECTOR = new DenseVector(new double[] { 1, 0, 1, 1, 0 });
  private Mapper.Context ctx;
  private MultiLabelVectorWritable instance;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();

    ctx = EasyMock.createMock(Mapper.Context.class);
    instance = new MultiLabelVectorWritable(VECTOR,
      new int[] {0});
  }
  
  @Test
  public void index() throws Exception {
    ctx.write(new IntWritable(0), new VectorWritable(VECTOR));
    EasyMock.replay(ctx);
    IndexInstancesMapper indexInstances = new IndexInstancesMapper();
    indexInstances.map(new IntWritable(-1), instance, ctx);
    EasyMock.verify(ctx);
  }
}
