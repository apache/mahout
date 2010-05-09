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

package org.apache.mahout.cf.taste.hadoop.pseudo;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.VLongWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

/**
 * Extracts and emits all user IDs from the users file, or input file.
 */
public final class UserIDsMapper extends MapReduceBase implements
    Mapper<LongWritable,Text,VLongWritable,NullWritable> {
  
  @Override
  public void map(LongWritable key,
                  Text value,
                  OutputCollector<VLongWritable,NullWritable> output,
                  Reporter reporter) throws IOException {
    String line = value.toString();
    int comma = line.indexOf(',');
    long userID = comma >= 0 ? Long.parseLong(line.substring(0, comma)) : Long.parseLong(line);
    output.collect(new VLongWritable(userID), NullWritable.get());
  }
  
}