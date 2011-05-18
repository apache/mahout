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
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.dirichlet.models.DistributionDescription;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.VectorWritable;

public class DirichletMapper extends Mapper<WritableComparable<?>, VectorWritable, Text, VectorWritable> {

  private DirichletClusterer clusterer;

  @Override
  protected void map(WritableComparable<?> key, VectorWritable v, Context context)
    throws IOException, InterruptedException {
    int k = clusterer.assignToModel(v);
    context.write(new Text(String.valueOf(k)), v);
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    DirichletState dirichletState = getDirichletState(context.getConfiguration());
    for (DirichletCluster cluster : dirichletState.getClusters()) {
      cluster.getModel().configure(context.getConfiguration());
    }
    clusterer = new DirichletClusterer(dirichletState);
    for (int i = 0; i < dirichletState.getNumClusters(); i++) {
      // write an empty vector to each clusterId so that all will be seen by a reducer
      // Reducers will ignore these points but every model will be processed by one of them
      context.write(new Text(Integer.toString(i)), new VectorWritable(new DenseVector(0)));
    }
  }

  public void setup(DirichletState state) {
    this.clusterer = new DirichletClusterer(state);
  }

  public static DirichletState getDirichletState(Configuration conf) {
    String statePath = conf.get(DirichletDriver.STATE_IN_KEY);
    String descriptionString = conf.get(DirichletDriver.MODEL_DISTRIBUTION_KEY);
    String numClusters = conf.get(DirichletDriver.NUM_CLUSTERS_KEY);
    String alpha0 = conf.get(DirichletDriver.ALPHA_0_KEY);

    DistributionDescription description = DistributionDescription.fromString(descriptionString);
    return loadState(conf, statePath, description, Double.parseDouble(alpha0), Integer.parseInt(numClusters));
  }

  protected static DirichletState loadState(Configuration conf,
                                            String statePath,
                                            DistributionDescription description,
                                            double alpha,
                                            int k) {
    DirichletState state = DirichletDriver.createState(description, k, alpha);
    Path path = new Path(statePath);
    for (Pair<Writable,DirichletCluster> record
         : new SequenceFileDirIterable<Writable,DirichletCluster>(path,
                                                                  PathType.LIST,
                                                                  PathFilters.logsCRCFilter(),
                                                                  conf)) {
      int index = Integer.parseInt(record.getFirst().toString());
      state.getClusters().set(index, record.getSecond());
    }
    // TODO: with more than one mapper, they will all have different mixtures. Will this matter?
    state.setMixture(UncommonDistributions.rDirichlet(state.totalCounts(), alpha));
    return state;
  }
}
