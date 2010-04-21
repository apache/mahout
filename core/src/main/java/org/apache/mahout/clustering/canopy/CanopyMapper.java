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

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class CanopyMapper extends MapReduceBase implements
    Mapper<WritableComparable<?>,VectorWritable,Text,VectorWritable> {
  
  private final List<Canopy> canopies = new ArrayList<Canopy>();
  
  private OutputCollector<Text,VectorWritable> outputCollector;
  
  private CanopyClusterer canopyClusterer;
  
  @Override
  public void map(WritableComparable<?> key,
                  VectorWritable point,
                  OutputCollector<Text,VectorWritable> output,
                  Reporter reporter) throws IOException {
    outputCollector = output;
    canopyClusterer.addPointToCanopies(point.get(), canopies, reporter);
  }
  
  @Override
  public void configure(JobConf job) {
    super.configure(job);
    canopyClusterer = new CanopyClusterer(job);
  }
  
  @Override
  public void close() throws IOException {
    for (Canopy canopy : canopies) {
      Vector centroid = canopy.computeCentroid();
      VectorWritable vw = new VectorWritable(centroid);
      outputCollector.collect(new Text("centroid"), vw);
    }
    super.close();
  }
  
}
