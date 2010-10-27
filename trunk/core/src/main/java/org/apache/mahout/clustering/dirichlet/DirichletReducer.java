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

import java.io.IOException;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.math.VectorWritable;

public class DirichletReducer extends Reducer<Text, VectorWritable, Text, DirichletCluster> {

  private DirichletClusterer clusterer;

  private Cluster[] newModels;

  public Cluster[] getNewModels() {
    return newModels;
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    try {
      clusterer = new DirichletClusterer(DirichletMapper.getDirichletState(context.getConfiguration()));
      this.newModels = (Cluster[]) clusterer.samplePosteriorModels();
    } catch (NumberFormatException e) {
      throw new IllegalStateException(e);
    } catch (SecurityException e) {
      throw new IllegalStateException(e);
    } catch (IllegalArgumentException e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  protected void reduce(Text key, Iterable<VectorWritable> values, Context context) throws IOException, InterruptedException {
    int k = Integer.parseInt(key.toString());
    Cluster model = newModels[k];
    for (VectorWritable value : values) {
      // only observe real points, not the empty placeholders emitted by each mapper
      if (value.get().size() > 0) {
        model.observe(value);
      }
    }
    DirichletCluster cluster = clusterer.updateCluster(model, k);
    context.write(new Text(String.valueOf(k)), cluster);
  }

  public void setup(DirichletState state) {
    clusterer = new DirichletClusterer(state);
    this.newModels = (Cluster[]) clusterer.samplePosteriorModels();
  }

}
