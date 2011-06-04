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

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.Iterator;

import com.google.common.base.Splitter;
import org.apache.mahout.clustering.ModelDistribution;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

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
  public ModelDistribution<VectorWritable> createModelDistribution() {
    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    AbstractVectorModelDistribution modelDistribution;
    try {
      Class<? extends AbstractVectorModelDistribution> cl = ccl.loadClass(modelFactory)
          .asSubclass(AbstractVectorModelDistribution.class);
      modelDistribution = cl.newInstance();

      Class<? extends Vector> vcl = ccl.loadClass(modelPrototype).asSubclass(Vector.class);
      Constructor<? extends Vector> v = vcl.getConstructor(int.class);
      modelDistribution.setModelPrototype(new VectorWritable(v.newInstance(prototypeSize)));

      if (modelDistribution instanceof DistanceMeasureClusterDistribution) {
        Class<? extends DistanceMeasure> measureCl = ccl.loadClass(distanceMeasure).asSubclass(DistanceMeasure.class);
        DistanceMeasure measure = measureCl.newInstance();
        ((DistanceMeasureClusterDistribution) modelDistribution).setMeasure(measure);
      }
    } catch (ClassNotFoundException cnfe) {
      throw new IllegalStateException(cnfe);
    } catch (NoSuchMethodException nsme) {
      throw new IllegalStateException(nsme);
    } catch (InstantiationException ie) {
      throw new IllegalStateException(ie);
    } catch (IllegalAccessException iae) {
      throw new IllegalStateException(iae);
    } catch (InvocationTargetException ite) {
      throw new IllegalStateException(ite);
    }
    return modelDistribution;
  }

  @Override
  public String toString() {
    return modelFactory + ',' + modelPrototype + ',' + distanceMeasure + ',' + prototypeSize;
  }

  public static DistributionDescription fromString(String s) {
    Iterator<String> tokens = Splitter.on(',').split(s).iterator();
    String modelFactory = tokens.next();
    String modelPrototype = tokens.next();
    String distanceMeasure = tokens.next();
    int prototypeSize = Integer.parseInt(tokens.next());
    return new DistributionDescription(modelFactory, modelPrototype, distanceMeasure, prototypeSize);
  }

}
