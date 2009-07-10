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

package org.apache.mahout.clustering.dirichlet;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.clustering.dirichlet.models.Model;
import org.apache.mahout.matrix.Vector;

import java.io.IOException;
import java.util.Iterator;

public class DirichletReducer extends MapReduceBase implements
    Reducer<Text, Vector, Text, DirichletCluster<Vector>> {

  DirichletState<Vector> state;

  public Model<Vector>[] newModels;

  @Override
  public void reduce(Text key, Iterator<Vector> values,
                     OutputCollector<Text, DirichletCluster<Vector>> output, Reporter reporter)
      throws IOException {
    int k = Integer.parseInt(key.toString());
    Model<Vector> model = newModels[k];
    while (values.hasNext()) {
      Vector v = values.next();
      model.observe(v);
    }
    model.computeParameters();
    DirichletCluster<Vector> cluster = state.clusters.get(k);
    cluster.setModel(model);
    output.collect(key, cluster);
  }

  public void configure(DirichletState<Vector> state) {
    this.state = state;
    this.newModels = state.modelFactory.sampleFromPosterior(state.getModels());
  }

  @Override
  public void configure(JobConf job) {
    super.configure(job);
    state = DirichletMapper.getDirichletState(job);
    this.newModels = state.modelFactory.sampleFromPosterior(state.getModels());
  }

}
