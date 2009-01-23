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

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.utils.StringUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

/**
 * Extract the attribute values from a dataline. Skip ignored attributes<br>
 * Input:<br>
 * <ul>
 * <li> LongWritable : data row index </li>
 * <li> Text : dataline </li>
 * </ul>
 * Output:<br>
 * <ul>
 * <li> LongWritable : attribute index.<br>
 * ignored attributes aren't taken into account when calculating this index.</li>
 * <li> Text : attribute value </li>
 * </ul>
 * 
 * See Descriptors, for more informations about the job parameter
 */
public class ToolMapper extends MapReduceBase implements
    Mapper<LongWritable, Text, LongWritable, Text> {

  public static final String ATTRIBUTES = "cdtool.attributes";

  private final List<String> attributes = new ArrayList<String>();

  private Descriptors descriptors;
  
  @Override
  public void configure(JobConf job) {
    super.configure(job);

    String descrs = job.get(ATTRIBUTES);

    if (descrs != null)
      configure((char[]) StringUtils.fromString(descrs));
  }

  void configure(char[] descriptors) {
    if (descriptors == null || descriptors.length == 0)
      throw new RuntimeException("Descriptors's array not found or is empty");

    this.descriptors = new Descriptors(descriptors);
  }

  @Override
  public void map(LongWritable key, Text value,
      OutputCollector<LongWritable, Text> output, Reporter reporter)
      throws IOException {
    extractAttributes(value, attributes);
    if (attributes.size() != descriptors.size())
      throw new RuntimeException(
          "Attributes number should be equal to the descriptors's array length");

    // output non ignored attributes
    for (int index = 0; index < attributes.size(); index++) {
      if (descriptors.isIgnored(index))
        continue;

      output.collect(new LongWritable(index), new Text(attributes.get(index)));
    }
  }

  /**
   * Extract attribute values from the input Text. The attributes are separated
   * by a colon ','. Skips ignored attributes.
   * 
   * @param value
   * @param attributes
   */
  static void extractAttributes(Text value, List<String> attributes) {
    StringTokenizer tokenizer = new StringTokenizer(value.toString(), ",");

    attributes.clear();
    while (tokenizer.hasMoreTokens()) {
      attributes.add(tokenizer.nextToken().trim());
    }
  }
}
