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

package org.apache.mahout.cf.taste.hadoop;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.hadoop.MathHelper;
import org.easymock.EasyMock;
import org.junit.Test;

public class MaybePruneRowsMapperTest extends TasteTestCase {

  @Test
  public void testPruning() throws Exception {
    Vector v1 = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    v1.set(1, 1);
    v1.set(3, 1);

    Vector v2 = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    v2.set(1, 1);
    v2.set(7, 1);

    Vector v3 = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    v3.set(1, 1);
    v3.set(5, 1);
    v3.set(9, 1);

    MaybePruneRowsMapper mapper = new MaybePruneRowsMapper();
    setField(mapper, "maxCooccurrences", 2);

    Mapper<VarLongWritable,VectorWritable, IntWritable, DistributedRowMatrix.MatrixEntryWritable>.Context ctx =
      EasyMock.createMock(Mapper.Context.class);
    Counter usedElementsCounter = EasyMock.createMock(Counter.class);
    Counter neglectedElementsCounter = EasyMock.createMock(Counter.class);

    ctx.write(EasyMock.eq(new IntWritable(1)), MathHelper.matrixEntryMatches(1, 123, 1));
    ctx.write(EasyMock.eq(new IntWritable(3)), MathHelper.matrixEntryMatches(3, 123, 1));
    EasyMock.expect(ctx.getCounter(MaybePruneRowsMapper.Elements.USED)).andReturn(usedElementsCounter);
    usedElementsCounter.increment(2);
    EasyMock.expect(ctx.getCounter(MaybePruneRowsMapper.Elements.NEGLECTED)).andReturn(neglectedElementsCounter);
    neglectedElementsCounter.increment(0);

    ctx.write(EasyMock.eq(new IntWritable(1)), MathHelper.matrixEntryMatches(1, 456, 1));
    ctx.write(EasyMock.eq(new IntWritable(7)), MathHelper.matrixEntryMatches(7, 456, 1));
    EasyMock.expect(ctx.getCounter(MaybePruneRowsMapper.Elements.USED)).andReturn(usedElementsCounter);
    usedElementsCounter.increment(2);
    EasyMock.expect(ctx.getCounter(MaybePruneRowsMapper.Elements.NEGLECTED)).andReturn(neglectedElementsCounter);
    neglectedElementsCounter.increment(0);

    ctx.write(EasyMock.eq(new IntWritable(5)), MathHelper.matrixEntryMatches(5, 789, 1));
    ctx.write(EasyMock.eq(new IntWritable(9)), MathHelper.matrixEntryMatches(9, 789, 1));
        EasyMock.expect(ctx.getCounter(MaybePruneRowsMapper.Elements.USED)).andReturn(usedElementsCounter);
    usedElementsCounter.increment(2);
    EasyMock.expect(ctx.getCounter(MaybePruneRowsMapper.Elements.NEGLECTED)).andReturn(neglectedElementsCounter);
    neglectedElementsCounter.increment(1);

    EasyMock.replay(ctx, usedElementsCounter, neglectedElementsCounter);

    mapper.map(new VarLongWritable(123L), new VectorWritable(v1), ctx);
    mapper.map(new VarLongWritable(456L), new VectorWritable(v2), ctx);
    mapper.map(new VarLongWritable(789L), new VectorWritable(v3), ctx);

    EasyMock.verify(ctx, usedElementsCounter, neglectedElementsCounter);
  }

}
