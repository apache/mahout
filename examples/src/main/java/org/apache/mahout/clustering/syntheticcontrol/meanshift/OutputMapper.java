package org.apache.mahout.clustering.syntheticcontrol.meanshift;

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

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.clustering.meanshift.MeanShiftCanopy;
import org.apache.mahout.matrix.Vector;

import java.io.IOException;

public class OutputMapper extends MapReduceBase implements
    Mapper<Text, Text, Text, Text> {

  int clusters = 0;

  public void map(Text key, Text values, OutputCollector<Text, Text> output,
      Reporter reporter) throws IOException {
    clusters++;
    String foo = values.toString();
    MeanShiftCanopy canopy = MeanShiftCanopy.decodeCanopy(foo);
    for (Vector point : canopy.getBoundPoints())
      output.collect(key, new Text(point.asFormatString()));
  }

  @Override
  public void close() throws IOException {
    System.out.println("+++ Clusters=" + clusters);
    super.close();
  }

}
