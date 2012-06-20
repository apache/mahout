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

package org.apache.mahout.clustering.dirichlet.models;

import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.clustering.ModelDistribution;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.google.common.base.Splitter;

/**
 * Simply describes parameters needs to create a {@link org.apache.mahout.clustering.ModelDistribution}.
 */
public final class DistributionDescription {
  
  private final String modelFactory;
  private final String modelPrototype;
  private final String distanceMeasure;
  private final int prototypeSize;
  
  public DistributionDescription(String modelFactory,
                                 String modelPrototype,
                                 String distanceMeasure,
                                 int prototypeSize) {
    this.modelFactory = modelFactory;
    this.modelPrototype = modelPrototype;
    this.distanceMeasure = distanceMeasure;
    this.prototypeSize = prototypeSize;
  }
  
  public String getModelFactory() {
    return modelFactory;
  }
  
  public String getModelPrototype() {
    return modelPrototype;
  }
  
  public String getDistanceMeasure() {
    return distanceMeasure;
  }
  
  public int getPrototypeSize() {
    return prototypeSize;
  }
  
  /**
   * Create an instance of AbstractVectorModelDistribution from the given command line arguments
   */
  public ModelDistribution<VectorWritable> createModelDistribution(Configuration conf) {
    AbstractVectorModelDistribution modelDistribution =
        ClassUtils.instantiateAs(modelFactory, AbstractVectorModelDistribution.class);

    Vector prototype = ClassUtils.instantiateAs(modelPrototype,
                                                Vector.class,
                                                new Class<?>[] {int.class},
                                                new Object[] {prototypeSize});
      
    modelDistribution.setModelPrototype(new VectorWritable(prototype));

    if (modelDistribution instanceof DistanceMeasureClusterDistribution) {
      DistanceMeasure measure = ClassUtils.instantiateAs(distanceMeasure, DistanceMeasure.class);
      measure.configure(conf);
      ((DistanceMeasureClusterDistribution) modelDistribution).setMeasure(measure);
    }

    return modelDistribution;
  }
  
  @Override
  public String toString() {
    return modelFactory + ',' + modelPrototype + ',' + distanceMeasure + ',' + prototypeSize;
  }
  
  public static DistributionDescription fromString(CharSequence s) {
    Iterator<String> tokens = Splitter.on(',').split(s).iterator();
    String modelFactory = tokens.next();
    String modelPrototype = tokens.next();
    String distanceMeasure = tokens.next();
    int prototypeSize = Integer.parseInt(tokens.next());
    return new DistributionDescription(modelFactory, modelPrototype, distanceMeasure, prototypeSize);
  }
  
}
