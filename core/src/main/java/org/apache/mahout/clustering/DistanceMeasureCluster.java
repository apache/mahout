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

package org.apache.mahout.clustering;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class DistanceMeasureCluster extends AbstractCluster {

  private DistanceMeasure measure;

  public DistanceMeasureCluster(Vector point, int id, DistanceMeasure measure) {
    super(point, id);
    this.measure = measure;
  }

  public DistanceMeasureCluster() {
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    String dm = in.readUTF();
    try {
      ClassLoader ccl = Thread.currentThread().getContextClassLoader();
      this.measure = ccl.loadClass(dm).asSubclass(DistanceMeasure.class).newInstance();
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    }
    super.readFields(in);
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeUTF(measure.getClass().getName());
    super.write(out);
  }

  @Override
  public double pdf(VectorWritable vw) {
    return Math.exp(-measure.distance(vw.get(), getCenter()));
  }

  @Override
  public Model<VectorWritable> sampleFromPosterior() {
    return new DistanceMeasureCluster(getCenter(), getId(), measure);
  }

  public DistanceMeasure getMeasure() {
    return measure;
  }

  /**
   * @param measure the measure to set
   */
  public void setMeasure(DistanceMeasure measure) {
    this.measure = measure;
  }

  @Override
  public String getIdentifier() {
    return "DMC:" + getId();
  }

}
