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

package org.apache.mahout.fpm.pfpgrowth.dataset;

import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.regex.Pattern;

import com.google.common.collect.Lists;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.common.StringTuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Splits the line using a {@link Pattern} and outputs key as given by the groupingFields
 * 
 */
public class KeyBasedStringTupleMapper extends Mapper<LongWritable,Text,Text,StringTuple> {
  
  private static final Logger log = LoggerFactory.getLogger(KeyBasedStringTupleMapper.class);
  
  private Pattern splitter;
  
  private int[] selectedFields;
  
  private int[] groupingFields;
  
  @Override
  protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
    String[] fields = splitter.split(value.toString());
    if (fields.length != 4) {
      log.info("{} {}", fields.length, value.toString());
      context.getCounter("Map", "ERROR").increment(1);
      return;
    }
    Collection<String> oKey = Lists.newArrayList();
    for (int groupingField : groupingFields) {
      oKey.add(fields[groupingField]);
      context.setStatus(fields[groupingField]);
    }
    
    List<String> oValue = Lists.newArrayList();
    for (int selectedField : selectedFields) {
      oValue.add(fields[selectedField]);
    }
    
    context.write(new Text(oKey.toString()), new StringTuple(oValue));
    
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Parameters params = new Parameters(context.getConfiguration().get("job.parameters", ""));
    splitter = Pattern.compile(params.get("splitPattern", "[ \t]*\t[ \t]*"));
    
    int selectedFieldCount = Integer.valueOf(params.get("selectedFieldCount", "0"));
    selectedFields = new int[selectedFieldCount];
    for (int i = 0; i < selectedFieldCount; i++) {
      selectedFields[i] = Integer.valueOf(params.get("field" + i, "0"));
    }
    
    int groupingFieldCount = Integer.valueOf(params.get("groupingFieldCount", "0"));
    groupingFields = new int[groupingFieldCount];
    for (int i = 0; i < groupingFieldCount; i++) {
      groupingFields[i] = Integer.valueOf(params.get("gfield" + i, "0"));
    }
    
  }
}
