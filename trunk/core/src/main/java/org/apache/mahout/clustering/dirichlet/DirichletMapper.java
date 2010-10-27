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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.JsonModelDistributionAdapter;
import org.apache.mahout.clustering.ModelDistribution;
import org.apache.mahout.clustering.dirichlet.models.AbstractVectorModelDistribution;
import org.apache.mahout.clustering.kmeans.OutputLogFilter;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.VectorWritable;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

public class DirichletMapper extends Mapper<WritableComparable<?>, VectorWritable, Text, VectorWritable> {

  private DirichletClusterer clusterer;

  @Override
  protected void map(WritableComparable<?> key, VectorWritable v, Context context) throws IOException, InterruptedException {
    int k = clusterer.assignToModel(v);
    context.write(new Text(String.valueOf(k)), v);
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    try {
      DirichletState dirichletState = getDirichletState(context.getConfiguration());
      clusterer = new DirichletClusterer(dirichletState);
      for (int i = 0; i < dirichletState.getNumClusters(); i++) {
        // write an empty vector to each clusterId so that all will be seen by a reducer
        // Reducers will ignore these points but every model will be processed by one of them
        context.write(new Text(Integer.toString(i)), new VectorWritable(new DenseVector(0)));
      }
    } catch (NumberFormatException e) {
      throw new IllegalStateException(e);
    } catch (SecurityException e) {
      throw new IllegalStateException(e);
    } catch (IllegalArgumentException e) {
      throw new IllegalStateException(e);
    }
  }

  public void setup(DirichletState state) {
    this.clusterer = new DirichletClusterer(state);
  }

  public static DirichletState getDirichletState(Configuration conf) {
    String statePath = conf.get(DirichletDriver.STATE_IN_KEY);
    String json = conf.get(DirichletDriver.MODEL_DISTRIBUTION_KEY);
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(ModelDistribution.class, new JsonModelDistributionAdapter());
    Gson gson = builder.create();
    ModelDistribution<VectorWritable> modelDistribution = gson.fromJson(json,
                                                                        AbstractVectorModelDistribution.MODEL_DISTRIBUTION_TYPE);
    String numClusters = conf.get(DirichletDriver.NUM_CLUSTERS_KEY);
    String alpha0 = conf.get(DirichletDriver.ALPHA_0_KEY);

    try {
      return loadState(conf, statePath, modelDistribution, Double.parseDouble(alpha0), Integer.parseInt(numClusters));
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

  protected static DirichletState loadState(Configuration conf,
                                            String statePath,
                                            ModelDistribution<VectorWritable> modelDistribution,
                                            double alpha,
                                            int k) throws IOException {
    DirichletState state = DirichletDriver.createState(modelDistribution, k, alpha);
    Path path = new Path(statePath);
    FileSystem fs = FileSystem.get(path.toUri(), conf);
    FileStatus[] status = fs.listStatus(path, new OutputLogFilter());
    for (FileStatus s : status) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, s.getPath(), conf);
      try {
        Writable key = new Text();
        DirichletCluster cluster = new DirichletCluster();
        while (reader.next(key, cluster)) {
          int index = Integer.parseInt(key.toString());
          state.getClusters().set(index, cluster);
          cluster = new DirichletCluster();
        }
      } finally {
        reader.close();
      }
    }
    // TODO: with more than one mapper, they will all have different mixtures. Will this matter?
    state.setMixture(UncommonDistributions.rDirichlet(state.totalCounts(), alpha));
    return state;
  }
}
