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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.kmeans.OutputLogFilter;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.VectorWritable;

public class DirichletMapper extends Mapper<WritableComparable<?>, VectorWritable, Text, VectorWritable> {

  private DirichletClusterer<VectorWritable> clusterer;

  @Override
  protected void map(WritableComparable<?> key, VectorWritable v, Context context) throws IOException, InterruptedException {
    int k = clusterer.assignToModel(v);
    context.write(new Text(String.valueOf(k)), v);
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    try {
      DirichletState<VectorWritable> dirichletState = getDirichletState(context.getConfiguration());
      clusterer = new DirichletClusterer<VectorWritable>(dirichletState);
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
    } catch (NoSuchMethodException e) {
      throw new IllegalStateException(e);
    } catch (InvocationTargetException e) {
      throw new IllegalStateException(e);
    }
  }

  public void setup(DirichletState<VectorWritable> state) {
    this.clusterer = new DirichletClusterer<VectorWritable>(state);
  }

  public static DirichletState<VectorWritable> getDirichletState(Configuration conf) throws NoSuchMethodException,
      InvocationTargetException {
    String statePath = conf.get(DirichletDriver.STATE_IN_KEY);
    String modelFactory = conf.get(DirichletDriver.MODEL_FACTORY_KEY);
    String modelPrototype = conf.get(DirichletDriver.MODEL_PROTOTYPE_KEY);
    String prototypeSize = conf.get(DirichletDriver.PROTOTYPE_SIZE_KEY);
    String numClusters = conf.get(DirichletDriver.NUM_CLUSTERS_KEY);
    String alpha0 = conf.get(DirichletDriver.ALPHA_0_KEY);

    try {
      return loadState(conf,
                       statePath,
                       modelFactory,
                       modelPrototype,
                       Double.parseDouble(alpha0),
                       Integer.parseInt(prototypeSize),
                       Integer.parseInt(numClusters));
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

  /**
   * @param conf
   * @param statePath
   * @param modelFactory
   * @param modelPrototype
   * @param alpha
   * @param pSize
   * @param k
   * @return
   * @throws ClassNotFoundException
   * @throws InstantiationException
   * @throws IllegalAccessException
   * @throws NoSuchMethodException
   * @throws InvocationTargetException
   * @throws IOException
   */
  protected static DirichletState<VectorWritable> loadState(Configuration conf,
                                                            String statePath,
                                                            String modelFactory,
                                                            String modelPrototype,
                                                            double alpha,
                                                            int pSize,
                                                            int k) throws ClassNotFoundException, InstantiationException,
      IllegalAccessException, NoSuchMethodException, InvocationTargetException, IOException {
    DirichletState<VectorWritable> state = DirichletDriver.createState(modelFactory, modelPrototype, pSize, k, alpha);
    Path path = new Path(statePath);
    FileSystem fs = FileSystem.get(path.toUri(), conf);
    FileStatus[] status = fs.listStatus(path, new OutputLogFilter());
    for (FileStatus s : status) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, s.getPath(), conf);
      try {
        Text key = new Text();
        DirichletCluster<VectorWritable> cluster = new DirichletCluster<VectorWritable>();
        while (reader.next(key, cluster)) {
          int index = Integer.parseInt(key.toString());
          state.getClusters().set(index, cluster);
          cluster = new DirichletCluster<VectorWritable>();
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
