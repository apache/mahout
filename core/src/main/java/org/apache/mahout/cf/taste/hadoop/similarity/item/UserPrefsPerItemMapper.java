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
import java.util.regex.Pattern;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.cf.taste.hadoop.EntityWritable;

/**
 * Read an entry from the preferences file and map it out with the item as key and the user with her preference
 * as value
 */
public final class UserPrefsPerItemMapper extends Mapper<LongWritable,Text, EntityWritable, EntityPrefWritable> {

  private static final Pattern COMMA = Pattern.compile(",");

  @Override
  protected void map(LongWritable key, Text value, Context context)
      throws IOException, InterruptedException {

    String[] tokens = COMMA.split(value.toString());

    long userID = Long.parseLong(tokens[0]);
    long itemID = Long.parseLong(tokens[1]);
    float pref = Float.parseFloat(tokens[2]);

    context.write(new EntityWritable(itemID), new EntityPrefWritable(userID,pref));
  }


}
