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

package org.apache.mahout.vectorizer.term;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * TextVectorizer Document Frequency Count Mapper. Outputs 1 for each feature
 */
public class TermDocumentCountMapper extends Mapper<WritableComparable<?>, VectorWritable, IntWritable, LongWritable> {

  private static final LongWritable ONE = new LongWritable(1);

  private static final IntWritable TOTAL_COUNT = new IntWritable(-1);

  private final IntWritable out = new IntWritable();

  @Override
  protected void map(WritableComparable<?> key, VectorWritable value, Context context)
    throws IOException, InterruptedException {
    Vector vector = value.get();
    for (Vector.Element e : vector.nonZeroes()) {
      out.set(e.index());
      context.write(out, ONE);
    }
    context.write(TOTAL_COUNT, ONE);
  }
}
