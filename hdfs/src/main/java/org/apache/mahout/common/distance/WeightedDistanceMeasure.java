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

package org.apache.mahout.common.distance;

import java.io.DataInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Collection;
import java.util.List;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.parameters.ClassParameter;
import org.apache.mahout.common.parameters.Parameter;
import org.apache.mahout.common.parameters.PathParameter;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/** Abstract implementation of DistanceMeasure with support for weights. */
public abstract class WeightedDistanceMeasure implements DistanceMeasure {
  
  private List<Parameter<?>> parameters;
  private Parameter<Path> weightsFile;
  private ClassParameter vectorClass;
  private Vector weights;
  
  @Override
  public void createParameters(String prefix, Configuration jobConf) {
    parameters = Lists.newArrayList();
    weightsFile = new PathParameter(prefix, "weightsFile", jobConf, null,
        "Path on DFS to a file containing the weights.");
    parameters.add(weightsFile);
    vectorClass = new ClassParameter(prefix, "vectorClass", jobConf, DenseVector.class,
        "Class<Vector> file specified in parameter weightsFile has been serialized with.");
    parameters.add(vectorClass);
  }
  
  @Override
  public Collection<Parameter<?>> getParameters() {
    return parameters;
  }
  
  @Override
  public void configure(Configuration jobConf) {
    if (parameters == null) {
      ParameteredGeneralizations.configureParameters(this, jobConf);
    }
    try {
      if (weightsFile.get() != null) {
        FileSystem fs = FileSystem.get(weightsFile.get().toUri(), jobConf);
        VectorWritable weights =
            ClassUtils.instantiateAs((Class<? extends VectorWritable>) vectorClass.get(), VectorWritable.class);
        if (!fs.exists(weightsFile.get())) {
          throw new FileNotFoundException(weightsFile.get().toString());
        }
        DataInputStream in = fs.open(weightsFile.get());
        try {
          weights.readFields(in);
        } finally {
          Closeables.close(in, true);
        }
        this.weights = weights.get();
      }
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }
  
  public Vector getWeights() {
    return weights;
  }
  
  public void setWeights(Vector weights) {
    this.weights = weights;
  }
  
}
