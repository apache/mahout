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

package org.apache.mahout.clustering.meanshift;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MeanShiftCanopyMapper extends MapReduceBase implements
    Mapper<WritableComparable<?>, MeanShiftCanopy, Text, MeanShiftCanopy> {

  private final List<MeanShiftCanopy> canopies = new ArrayList<MeanShiftCanopy>();

  private OutputCollector<Text, MeanShiftCanopy> output;

  @Override
  public void map(WritableComparable<?> key, MeanShiftCanopy canopy,
                  OutputCollector<Text, MeanShiftCanopy> output, Reporter reporter)
      throws IOException {
    this.output = output;
    MeanShiftCanopy.mergeCanopy(canopy.shallowCopy(), canopies);
  }

  @Override
  public void close() throws IOException {
    for (MeanShiftCanopy canopy : canopies) {
      canopy.shiftToMean();
      output.collect(new Text("canopy"), canopy);
    }
    super.close();
  }

  @Override
  public void configure(JobConf job) {
    super.configure(job);
    MeanShiftCanopy.configure(job);
  }

}
