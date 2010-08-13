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

package org.apache.mahout.cf.taste.hadoop.similarity.item;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;

/**
 * creates an item-user-matrix entry from a preference, replacing userID and itemID with int indices
 */
public class PrefsToItemUserMatrixMapper
    extends Mapper<LongWritable,Text,VarIntWritable,DistributedRowMatrix.MatrixEntryWritable> {

  public static final String BOOLEAN_DATA = PrefsToItemUserMatrixMapper.class.getName() + ".booleanData";

  private boolean booleanData;
  
  @Override
  protected void setup(Context ctx) throws IOException, InterruptedException {
    booleanData = ctx.getConfiguration().getBoolean(BOOLEAN_DATA, false);
  }

  @Override
  protected void map(LongWritable key, Text value, Context ctx)
      throws IOException, InterruptedException {

    String[] tokens = TasteHadoopUtils.splitPrefTokens(value.toString());
    long userID = Long.parseLong(tokens[0]);
    long itemID = Long.parseLong(tokens[1]);

    boolean treatAsBoolean = booleanData || tokens.length < 3;
    float prefValue = treatAsBoolean ? 1.0f : Float.parseFloat(tokens[2]);

    int row = TasteHadoopUtils.idToIndex(itemID);
    int column = TasteHadoopUtils.idToIndex(userID);

    DistributedRowMatrix.MatrixEntryWritable entry = new DistributedRowMatrix.MatrixEntryWritable();
    entry.setRow(row);
    entry.setCol(column);
    entry.setVal(prefValue);

    ctx.write(new VarIntWritable(row), entry);
  }

}
