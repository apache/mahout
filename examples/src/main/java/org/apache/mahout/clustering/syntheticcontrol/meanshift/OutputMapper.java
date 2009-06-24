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

package org.apache.mahout.clustering.syntheticcontrol.meanshift;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.clustering.meanshift.MeanShiftCanopy;
import org.apache.mahout.matrix.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class OutputMapper extends MapReduceBase implements
    Mapper<Text, MeanShiftCanopy, Text, Text> {

  private static final Logger log = LoggerFactory.getLogger(OutputMapper.class);

  private int clusters = 0;

  @Override
  public void map(Text key, MeanShiftCanopy canopy, OutputCollector<Text, Text> output,
      Reporter reporter) throws IOException {
    clusters++;
    for (Vector point : canopy.getBoundPoints())
      output.collect(key, new Text(point.asFormatString()));
  }

  @Override
  public void close() throws IOException {
    log.info("+++ Clusters={}", clusters);
    super.close();
  }

}
