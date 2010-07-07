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
import java.lang.reflect.InvocationTargetException;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.clustering.dirichlet.models.Model;
import org.apache.mahout.math.VectorWritable;

public class DirichletReducer extends Reducer<Text, VectorWritable, Text, DirichletCluster<VectorWritable>> {

  private DirichletState<VectorWritable> state;

  private Model<VectorWritable>[] newModels;

  public Model<VectorWritable>[] getNewModels() {
    return newModels;
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    try {
      state = DirichletMapper.getDirichletState(context.getConfiguration());
    } catch (NumberFormatException e) {
      throw new IllegalStateException(e);
    } catch (SecurityException e) {
      throw new IllegalStateException(e);
    } catch (IllegalArgumentException e) {
      throw new IllegalStateException(e);
    } catch (NoSuchMethodException e) {
      throw new IllegalStateException(e);
    } catch (InvocationTargetException e) {
      throw new IllegalStateException(e);
    }
    this.newModels = state.getModelFactory().sampleFromPosterior(state.getModels());
  }

  @Override
  protected void reduce(Text key, Iterable<VectorWritable> values, Context context) throws IOException, InterruptedException {
    int k = Integer.parseInt(key.toString());
    Model<VectorWritable> model = newModels[k];
    for (VectorWritable value : values) {
      model.observe(value);
    }
    model.computeParameters();
    DirichletCluster<VectorWritable> cluster = state.getClusters().get(k);
    cluster.setModel(model);
  }

  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    for (int i = 0; i < state.getNumClusters(); i++) {
      DirichletCluster<VectorWritable> cluster = state.getClusters().get(i);
      context.write(new Text(String.valueOf(i)), cluster);
    }
    super.cleanup(context);
  }

  public void setup(DirichletState<VectorWritable> state) {
    this.state = state;
    this.newModels = state.getModelFactory().sampleFromPosterior(state.getModels());
  }

}
