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
import java.util.List;

import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.Vector;

public final class PartialMultiplyMapper extends MapReduceBase implements
    Mapper<VarIntWritable,VectorAndPrefsWritable,VarLongWritable,VectorWritable> {

  @Override
  public void map(VarIntWritable key,
                  VectorAndPrefsWritable vectorAndPrefsWritable,
                  OutputCollector<VarLongWritable, VectorWritable> output,
                  Reporter reporter) throws IOException {

    int itemIndex = key.get();

    Vector cooccurrenceColumn = vectorAndPrefsWritable.getVector();
    List<Long> userIDs = vectorAndPrefsWritable.getUserIDs();
    List<Float> prefValues = vectorAndPrefsWritable.getValues();

    VarLongWritable userIDWritable = new VarLongWritable();

    // These single-element vectors ensure that each user will not be recommended
    // this item
    Vector excludeVector = new RandomAccessSparseVector(Integer.MAX_VALUE, 1);
    excludeVector.set(itemIndex, Double.NaN);
    VectorWritable excludeWritable = new VectorWritable(excludeVector);
    excludeWritable.setWritesLaxPrecision(true);
    for (long userID : userIDs) {
      userIDWritable.set(userID);
      output.collect(userIDWritable, excludeWritable);
    }

    VectorWritable vectorWritable = new VectorWritable();
    vectorWritable.setWritesLaxPrecision(true);

    for (int i = 0; i < userIDs.size(); i++) {
      long userID = userIDs.get(i);
      float prefValue = prefValues.get(i);
      if (!Float.isNaN(prefValue)) {
        Vector partialProduct = prefValue == 1.0f ? cooccurrenceColumn : 
            cooccurrenceColumn.times(prefValue);
        userIDWritable.set(userID);
        vectorWritable.set(partialProduct);
        output.collect(userIDWritable, vectorWritable);
      }
    }

  }

}