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

package org.apache.mahout.cf.taste.hadoop.item;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * maps a row of the similarity matrix to a {@link VectorOrPrefWritable}
 * 
 * actually a column from that matrix has to be used but as the similarity matrix is symmetric, 
 * we can use a row instead of having to transpose it
 */
public final class SimilarityMatrixRowWrapperMapper extends
    Mapper<IntWritable,VectorWritable,VarIntWritable,VectorOrPrefWritable> {

  private final VarIntWritable index = new VarIntWritable();
  private final VectorOrPrefWritable vectorOrPref = new VectorOrPrefWritable();

  @Override
  protected void map(IntWritable key,
                     VectorWritable value,
                     Context context) throws IOException, InterruptedException {
    Vector similarityMatrixRow = value.get();
    /* remove self similarity */
    similarityMatrixRow.set(key.get(), Double.NaN);

    index.set(key.get());
    vectorOrPref.set(similarityMatrixRow);

    context.write(index, vectorOrPref);
  }

}
