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

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.clustering.meanshift.MeanShiftCanopy;
import org.apache.mahout.matrix.DenseVector;
import org.apache.mahout.matrix.Vector;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class InputMapper extends MapReduceBase implements
    Mapper<LongWritable, Text, Text, MeanShiftCanopy> {

  @Override
  public void map(LongWritable key, Text values,
      OutputCollector<Text, MeanShiftCanopy> output, Reporter reporter) throws IOException {
    String[] numbers = values.toString().split(" ");
    // sometimes there are multiple separator spaces
    List<Double> doubles = new ArrayList<Double>();
    for (String value : numbers) {
      if (value.length() > 0)
        doubles.add(Double.valueOf(value));
    }
    Vector point = new DenseVector(doubles.size());
    int index = 0;
    for (Double d : doubles)
      point.set(index++, d);
    MeanShiftCanopy canopy = new MeanShiftCanopy(point);
    output.collect(new Text(), canopy);
  }

}
