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
package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

import java.io.IOException;

/**
 * Performs L1 normalization of input vectors.
 */
public class CVB0TopicTermVectorNormalizerMapper extends
    Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

  @Override
  protected void map(IntWritable key, VectorWritable value, Context context) throws IOException,
      InterruptedException {
    value.get().assign(Functions.div(value.get().norm(1.0)));
    context.write(key, value);
  }
}
