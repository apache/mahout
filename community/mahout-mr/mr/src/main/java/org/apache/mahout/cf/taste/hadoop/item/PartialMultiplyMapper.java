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

import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;

/**
 * maps similar items and their preference values per user
 */
public final class PartialMultiplyMapper extends
    Mapper<VarIntWritable,VectorAndPrefsWritable,VarLongWritable,PrefAndSimilarityColumnWritable> {

  private final VarLongWritable userIDWritable = new VarLongWritable();
  private final PrefAndSimilarityColumnWritable prefAndSimilarityColumn = new PrefAndSimilarityColumnWritable();

  @Override
  protected void map(VarIntWritable key,
                     VectorAndPrefsWritable vectorAndPrefsWritable,
                     Context context) throws IOException, InterruptedException {

    Vector similarityMatrixColumn = vectorAndPrefsWritable.getVector();
    List<Long> userIDs = vectorAndPrefsWritable.getUserIDs();
    List<Float> prefValues = vectorAndPrefsWritable.getValues();

    for (int i = 0; i < userIDs.size(); i++) {
      long userID = userIDs.get(i);
      float prefValue = prefValues.get(i);
      if (!Float.isNaN(prefValue)) {
        prefAndSimilarityColumn.set(prefValue, similarityMatrixColumn);
        userIDWritable.set(userID);
        context.write(userIDWritable, prefAndSimilarityColumn);
      }
    }
  }

}
