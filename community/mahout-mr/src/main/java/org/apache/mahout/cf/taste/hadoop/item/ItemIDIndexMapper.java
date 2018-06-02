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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.hadoop.ToEntityPrefsMapper;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;

public final class ItemIDIndexMapper extends
    Mapper<LongWritable,Text, VarIntWritable, VarLongWritable> {

  private boolean transpose;

  private final VarIntWritable indexWritable = new VarIntWritable();
  private final VarLongWritable itemIDWritable = new VarLongWritable();

  @Override
  protected void setup(Context context) {
    Configuration jobConf = context.getConfiguration();
    transpose = jobConf.getBoolean(ToEntityPrefsMapper.TRANSPOSE_USER_ITEM, false);
  }
  
  @Override
  protected void map(LongWritable key,
                     Text value,
                     Context context) throws IOException, InterruptedException {
    String[] tokens = TasteHadoopUtils.splitPrefTokens(value.toString());
    long itemID = Long.parseLong(tokens[transpose ? 0 : 1]);
    int index = TasteHadoopUtils.idToIndex(itemID);
    indexWritable.set(index);
    itemIDWritable.set(itemID);
    context.write(indexWritable, itemIDWritable);
  }  
}
