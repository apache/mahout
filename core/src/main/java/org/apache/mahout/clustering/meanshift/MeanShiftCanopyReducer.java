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

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class MeanShiftCanopyReducer extends MapReduceBase implements
    Reducer<Text, MeanShiftCanopy, Text, MeanShiftCanopy> {

  private final List<MeanShiftCanopy> canopies = new ArrayList<MeanShiftCanopy>();

  private boolean allConverged = true;

  private JobConf conf;

  @Override
  public void reduce(Text key, Iterator<MeanShiftCanopy> values,
                     OutputCollector<Text, MeanShiftCanopy> output, Reporter reporter)
      throws IOException {

    while (values.hasNext()) {
      MeanShiftCanopy canopy = values.next();
      MeanShiftCanopy.mergeCanopy(canopy.shallowCopy(), canopies);
    }

    for (MeanShiftCanopy canopy : canopies) {
      allConverged = canopy.shiftToMean() && allConverged;
      output.collect(new Text(canopy.getIdentifier()), canopy);
    }

  }

  @Override
  public void configure(JobConf job) {
    super.configure(job);
    this.conf = job;
    MeanShiftCanopy.configure(job);
  }

  @Override
  public void close() throws IOException {
    if (allConverged) {
      Path path = new Path(conf.get(MeanShiftCanopy.CONTROL_PATH_KEY));
      FileSystem.get(conf).createNewFile(path);
    }
    super.close();
  }

}
