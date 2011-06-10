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

package org.apache.mahout.ga.watchmaker.cd.tool;

import java.io.IOException;
import java.util.List;
import java.util.regex.Pattern;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

/**
 * Extract the attribute values from a dataline. Skip ignored attributes<br>
 * Input:<br>
 * <ul>
 * <li>LongWritable : data row index</li>
 * <li>Text : dataline</li>
 * </ul>
 * Output:<br>
 * <ul>
 * <li>LongWritable : attribute index.<br>
 * ignored attributes aren't taken into account when calculating this index.</li>
 * <li>Text : attribute value</li>
 * </ul>
 * 
 * See Descriptors, for more information about the job parameter
 */
public class ToolMapper extends Mapper<LongWritable, Text, LongWritable, Text> {

  public static final String ATTRIBUTES = "cdtool.attributes";
  private static final Pattern COMMA = Pattern.compile(",");

  private Descriptors descriptors;

  @Override
  protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
    List<String> attributes = extractAttributes(value);
    Preconditions.checkArgument(attributes.size() == descriptors.size(),
        "Attributes number should be equal to the descriptors's array length");

    // output non ignored attributes
    for (int index = 0; index < attributes.size(); index++) {
      if (descriptors.isIgnored(index)) {
        continue;
      }

      context.write(new LongWritable(index), new Text(attributes.get(index)));
    }
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    String descriptors = context.getConfiguration().get(ATTRIBUTES);

    if (descriptors != null) {
      configure(descriptors.toCharArray());
    }
  }

  void configure(char[] descriptors) {
    Preconditions.checkArgument(descriptors != null && descriptors.length > 0, "descriptors null or empty");
    this.descriptors = new Descriptors(descriptors);
  }

  /**
   * Extract attribute values from the input Text. The attributes are separated by a colon ','. Skips ignored
   * attributes.
   */
  static List<String> extractAttributes(Text value) {
    List<String> result = Lists.newArrayList();
    for (String token : COMMA.split(value.toString())) {
      result.add(token.trim());
    }
    return result;
  }
}
