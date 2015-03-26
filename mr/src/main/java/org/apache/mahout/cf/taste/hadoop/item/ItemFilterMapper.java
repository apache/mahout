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

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.VarLongWritable;

import java.io.IOException;
import java.util.regex.Pattern;

/**
 * map out all user/item pairs to filter, keyed by the itemID
 */
public class ItemFilterMapper extends Mapper<LongWritable,Text,VarLongWritable,VarLongWritable> {

  private static final Pattern SEPARATOR = Pattern.compile("[\t,]");

  private final VarLongWritable itemIDWritable = new VarLongWritable();
  private final VarLongWritable userIDWritable = new VarLongWritable();

  @Override
  protected void map(LongWritable key, Text line, Context ctx) throws IOException, InterruptedException {
    String[] tokens = SEPARATOR.split(line.toString());
    long userID = Long.parseLong(tokens[0]);
    long itemID = Long.parseLong(tokens[1]);
    itemIDWritable.set(itemID);
    userIDWritable.set(userID);
    ctx.write(itemIDWritable, userIDWritable);
  }
}
