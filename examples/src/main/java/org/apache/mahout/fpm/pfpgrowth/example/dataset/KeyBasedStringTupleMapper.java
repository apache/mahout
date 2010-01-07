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

package org.apache.mahout.fpm.pfpgrowth.example.dataset;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.common.StringTuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class KeyBasedStringTupleMapper extends Mapper<LongWritable, Text, Text, StringTuple> {

  private static final Logger log = LoggerFactory.getLogger(KeyBasedStringTupleMapper.class);

  private Pattern splitter = null;

  private int[] selectedFields = null;

  private int[] groupingFields = null;

  protected void map(LongWritable key, Text value, Context context) throws IOException,
      InterruptedException {
    String[] fields = splitter.split(value.toString());
    if (fields.length != 4) {
      log.info("{} {}", fields.length, value.toString());
      context.getCounter("Map", "ERROR").increment(1);
      return;
    }
    List<String> oKey = new ArrayList<String>();
    for (int i = 0, groupingFieldCount = groupingFields.length; i < groupingFieldCount; i++) {
      oKey.add(fields[groupingFields[i]]);
      context.setStatus(fields[groupingFields[i]]);
    }

    List<String> oValue = new ArrayList<String>();
    for (int i = 0, selectedFieldCount = selectedFields.length; i < selectedFieldCount; i++) {
      oValue.add(fields[selectedFields[i]]);
    }

    context.write(new Text(oKey.toString()), new StringTuple(oValue));

  }

  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Parameters params = Parameters.fromString(context.getConfiguration().get("job.parameters", ""));
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
