/*
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

package org.apache.mahout.common;

import org.apache.hadoop.io.IntWritable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Assert;
import org.junit.Test;

public class DummyRecordWriterTest {

  @Test
  public void testWrite() {
    DummyRecordWriter<IntWritable, VectorWritable> writer = 
        new DummyRecordWriter<IntWritable, VectorWritable>();
    IntWritable reusableIntWritable = new IntWritable();
    VectorWritable reusableVectorWritable = new VectorWritable();
    reusableIntWritable.set(0);
    reusableVectorWritable.set(new DenseVector(new double[] { 1, 2, 3 }));
    writer.write(reusableIntWritable, reusableVectorWritable);
    reusableIntWritable.set(1);
    reusableVectorWritable.set(new DenseVector(new double[] { 4, 5, 6 }));
    writer.write(reusableIntWritable, reusableVectorWritable);

    Assert.assertEquals(
        "The writer must remember the two keys that is written to it", 2,
        writer.getKeys().size());
  }
}
