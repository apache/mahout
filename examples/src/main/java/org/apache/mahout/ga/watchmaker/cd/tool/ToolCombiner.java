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
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.utils.StringUtils;

import java.io.IOException;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

/**
 * Combines attribute values into a String.<br>
 * <ul>
 * <li>For Numerical attributes, the string contains the min and max values
 * found.</li>
 * <li>For Categorical attributes, the string contains the distinct values
 * found.</li>
 * </ul>
 * 
 * See Descriptors, for more informations about the job parameter
 */
public class ToolCombiner extends MapReduceBase implements
    Reducer<LongWritable, Text, LongWritable, Text> {

  private final Set<String> distinct = new HashSet<String>();

  private Descriptors descriptors;

  @Override
  public void configure(JobConf job) {
    super.configure(job);

    String descriptors = job.get(ToolMapper.ATTRIBUTES);

    if (descriptors != null)
      configure((char[]) StringUtils.fromString(descriptors));
  }

  void configure(char[] descriptors) {
    if (descriptors == null || descriptors.length == 0)
      throw new RuntimeException("Descriptors's array not found or is empty");

    this.descriptors = new Descriptors(descriptors);
  }

  @Override
  public void reduce(LongWritable key, Iterator<Text> values,
      OutputCollector<LongWritable, Text> output, Reporter reporter)
      throws IOException {
    output.collect(key, new Text(createDescription((int) key.get(), values)));
  }

  /**
   * Generate a String description for a given attribute from its available
   * values.
   * 
   * @param index attribute index
   * @param values available values
   * @return
   * @throws RuntimeException if the attribute should be ignored.
   */
  String createDescription(int index, Iterator<Text> values) {
    if (descriptors.isNominal(index))
      return nominalDescription(values);
    else if (descriptors.isNumerical(index))
      return numericalDescription(values);
    else
      throw new RuntimeException(
          "An ignored attribute should never reach the Combiner");
  }

  String nominalDescription(Iterator<Text> values) {
    // distinct values
    distinct.clear();
    while (values.hasNext()) {
      distinct.add(values.next().toString());
    }

    return DescriptionUtils.createNominalDescription(distinct);
  }

  static String numericalDescription(Iterator<Text> values) {
    double min = Double.POSITIVE_INFINITY;
    double max = Double.NEGATIVE_INFINITY;

    while (values.hasNext()) {
      double value = Double.parseDouble(values.next().toString());
      if (value < min)
        min = value;
      else if (value > max)
        max = value;
    }

    return DescriptionUtils.createNumericalDescription(min, max);
  }
}
