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

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.matrix.AbstractVector;
import org.apache.mahout.matrix.Vector;

public class MeanShiftCanopyCombiner extends MapReduceBase implements
    Reducer<Text, WritableComparable<?>, Text, WritableComparable<?>> {

  @Override
  public void reduce(Text key, Iterator<WritableComparable<?>> values,
      OutputCollector<Text, WritableComparable<?>> output, Reporter reporter)
      throws IOException {
    MeanShiftCanopy canopy = new MeanShiftCanopy(key.toString());

    while (values.hasNext()) {
      Writable value = values.next();
      String valueStr = value.toString();
      if (valueStr.startsWith("new"))
        canopy.init(MeanShiftCanopy.decodeCanopy(valueStr.substring(4)));
      else if (valueStr.startsWith("merge"))
        canopy.merge(MeanShiftCanopy.decodeCanopy(valueStr.substring(6)));
      else {
        Vector formatString = AbstractVector.decodeVector(new Text(valueStr));
        int number = Integer.parseInt(valueStr.substring(valueStr.indexOf(":=:") + 3));
        canopy.addPoints(formatString, number);
      }
    }
    // Combiner may see situations where a canopy touched others in the mapper
    // before it was merged. This causes points to be added to it in the
    // combiner, but since the canopy was merged it has no center. Ignore
    // these cases.
    if (canopy.getCenter() != null) {
      canopy.shiftToMean();
      output.collect(new Text("canopy"), new Text(MeanShiftCanopy
          .formatCanopy(canopy)));
    }

  }

  @Override
  public void configure(JobConf job) {
    super.configure(job);
    MeanShiftCanopy.configure(job);
  }
}
