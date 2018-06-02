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

package org.apache.mahout.ep;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import com.google.common.base.Preconditions;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.sgd.PolymorphicWritable;
import org.apache.mahout.math.function.DoubleFunction;

/**
 * Provides coordinate tranformations so that evolution can proceed on the entire space of
 * reals but have the output limited and squished in convenient (and safe) ways.
 */
public abstract class Mapping extends DoubleFunction implements Writable {

  private Mapping() {
  }

  public static final class SoftLimit extends Mapping {
    private double min;
    private double max;
    private double scale;

    public SoftLimit() {
    }

    private SoftLimit(double min, double max, double scale) {
      this.min = min;
      this.max = max;
      this.scale = scale;
    }

    @Override
    public double apply(double v) {
      return min + (max - min) * 1 / (1 + Math.exp(-v * scale));
    }

    @Override
    public void write(DataOutput out) throws IOException {
      out.writeDouble(min);
      out.writeDouble(max);
      out.writeDouble(scale);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
      min = in.readDouble();
      max = in.readDouble();
      scale = in.readDouble();
    }
  }

  public static final class LogLimit extends Mapping {
    private Mapping wrapped;

    public LogLimit() {
    }

    private LogLimit(double low, double high) {
      wrapped = softLimit(Math.log(low), Math.log(high));
    }

    @Override
    public double apply(double v) {
      return Math.exp(wrapped.apply(v));
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
      PolymorphicWritable.write(dataOutput, wrapped);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
      wrapped = PolymorphicWritable.read(in, Mapping.class);
    }
  }

  public static final class Exponential extends Mapping {
    private double scale;

    public Exponential() {
    }

    private Exponential(double scale) {
      this.scale = scale;
    }

    @Override
    public double apply(double v) {
      return Math.exp(v * scale);
    }

    @Override
    public void write(DataOutput out) throws IOException {
      out.writeDouble(scale);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
      scale = in.readDouble();
    }
  }

  public static final class Identity extends Mapping {
    @Override
    public double apply(double v) {
      return v;
    }

    @Override
    public void write(DataOutput dataOutput) {
      // stateless
    }

    @Override
    public void readFields(DataInput dataInput) {
      // stateless
    }
  }

  /**
   * Maps input to the open interval (min, max) with 0 going to the mean of min and
   * max.  When scale is large, a larger proportion of values are mapped to points
   * near the boundaries.  When scale is small, a larger proportion of values are mapped to
   * points well within the boundaries.
   * @param min The largest lower bound on values to be returned.
   * @param max The least upper bound on values to be returned.
   * @param scale  Defines how sharp the boundaries are.
   * @return A mapping that satisfies the desired constraint.
   */
  public static Mapping softLimit(double min, double max, double scale) {
    return new SoftLimit(min, max, scale);
  }

  /**
   * Maps input to the open interval (min, max) with 0 going to the mean of min and
   * max.  When scale is large, a larger proportion of values are mapped to points
   * near the boundaries.
   * @see #softLimit(double, double, double)
   * @param min The largest lower bound on values to be returned.
   * @param max The least upper bound on values to be returned.
   * @return A mapping that satisfies the desired constraint.
   */
  public static Mapping softLimit(double min, double max) {
    return softLimit(min, max, 1);
  }

  /**
   * Maps input to positive values in the open interval (min, max) with
   * 0 going to the geometric mean.  Near the geometric mean, values are
   * distributed roughly geometrically.
   * @param low   The largest lower bound for output results.  Must be >0.
   * @param high  The least upper bound for output results.  Must be >0.
   * @return A mapped value.
   */
  public static Mapping logLimit(double low, double high) {
    Preconditions.checkArgument(low > 0, "Lower bound for log limit must be > 0 but was %f", low);
    Preconditions.checkArgument(high > 0, "Upper bound for log limit must be > 0 but was %f", high);
    return new LogLimit(low, high);
  }

  /**
   * Maps results to positive values.
   * @return A positive value.
   */
  public static Mapping exponential() {
    return exponential(1);
  }

  /**
   * Maps results to positive values.
   * @param scale  If large, then large values are more likely.
   * @return A positive value.
   */
  public static Mapping exponential(double scale) {
    return new Exponential(scale);
  }

  /**
   * Maps results to themselves.
   * @return The original value.
   */
  public static Mapping identity() {
    return new Identity();
  }
}
