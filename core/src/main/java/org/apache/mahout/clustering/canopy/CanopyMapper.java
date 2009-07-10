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

package org.apache.mahout.clustering.canopy;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.matrix.Vector;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CanopyMapper extends MapReduceBase implements
    Mapper<WritableComparable<?>, Vector, Text, Vector> {

  private final List<Canopy> canopies = new ArrayList<Canopy>();

  private OutputCollector<Text, Vector> outputCollector;

  @Override
  public void map(WritableComparable<?> key, Vector point,
                  OutputCollector<Text, Vector> output, Reporter reporter) throws IOException {
    outputCollector = output;
    Canopy.addPointToCanopies(point, canopies);
  }

  @Override
  public void configure(JobConf job) {
    super.configure(job);
    Canopy.configure(job);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.hadoop.mapred.MapReduceBase#close()
   */
  @Override
  public void close() throws IOException {
    for (Canopy canopy : canopies) {
      Vector centroid = canopy.computeCentroid();
      outputCollector.collect(new Text("centroid"), centroid);
    }
    super.close();
  }

}
