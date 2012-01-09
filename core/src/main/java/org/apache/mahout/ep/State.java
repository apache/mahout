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

import com.google.common.collect.Lists;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.sgd.PolymorphicWritable;
import org.apache.mahout.common.RandomUtils;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Records evolutionary state and provides a mutation operation for recorded-step meta-mutation.
 *
 * You provide the payload, this class provides the mutation operations.  During mutation,
 * the payload is copied and after the state variables are changed, they are passed to the
 * payload.
 *
 * Parameters are internally mutated in a state space that spans all of R^n, but parameters
 * passed to the payload are transformed as specified by a call to setMap().  The default
 * mapping is the identity map, but uniform-ish or exponential-ish coverage of a range are
 * also supported.
 *
 * More information on the underlying algorithm can be found in the following paper
 *
 * http://arxiv.org/abs/0803.3838
 *
 * @see Mapping
 */
public class State<T extends Payload<U>, U> implements Comparable<State<T, U>>, Writable {

  // object count is kept to break ties in comparison.
  private static final AtomicInteger OBJECT_COUNT = new AtomicInteger();

  private int id = OBJECT_COUNT.getAndIncrement();
  private Random gen = RandomUtils.getRandom();
  // current state
  private double[] params;
  // mappers to transform state
  private Mapping[] maps;
  // omni-directional mutation
  private double omni;
  // directional mutation
  private double[] step;
  // current fitness value
  private double value;
  private T payload;

  public State() {
  }

  /**
   * Invent a new state with no momentum (yet).
   */
  public State(double[] x0, double omni) {
    params = Arrays.copyOf(x0, x0.length);
    this.omni = omni;
    step = new double[params.length];
    maps = new Mapping[params.length];
  }

  /**
   * Deep copies a state, useful in mutation.
   */
  public State<T, U> copy() {
    State<T, U> r = new State<T, U>();
    r.params = Arrays.copyOf(this.params, this.params.length);
    r.omni = this.omni;
    r.step = Arrays.copyOf(this.step, this.step.length);
    r.maps = Arrays.copyOf(this.maps, this.maps.length);
    if (this.payload != null) {
      r.payload = (T) this.payload.copy();
    }
    r.gen = this.gen;
    return r;
  }

  /**
   * Clones this state with a random change in position.  Copies the payload and
   * lets it know about the change.
   *
   * @return A new state.
   */
  public State<T, U> mutate() {
    double sum = 0;
    for (double v : step) {
      sum += v * v;
    }
    sum = Math.sqrt(sum);
    double lambda = 1 + gen.nextGaussian();

    State<T, U> r = this.copy();
    double magnitude = 0.9 * omni + sum / 10;
    r.omni = magnitude * -Math.log1p(-gen.nextDouble());
    for (int i = 0; i < step.length; i++) {
      r.step[i] = lambda * step[i] + r.omni * gen.nextGaussian();
      r.params[i] += r.step[i];
    }
    if (this.payload != null) {
      r.payload.update(r.getMappedParams());
    }
    return r;
  }

  /**
   * Defines the transformation for a parameter.
   * @param i Which parameter's mapping to define.
   * @param m The mapping to use.
   * @see org.apache.mahout.ep.Mapping
   */
  public void setMap(int i, Mapping m) {
    maps[i] = m;
  }

  /**
   * Returns a transformed parameter.
   * @param i  The parameter to return.
   * @return The value of the parameter.
   */
  public double get(int i) {
    Mapping m = maps[i];
    return m == null ? params[i] : m.apply(params[i]);
  }

  public int getId() {
    return id;
  }

  public double[] getParams() {
    return params;
  }

  public Mapping[] getMaps() {
    return maps;
  }

  /**
   * Returns all the parameters in mapped form.
   * @return An array of parameters.
   */
  public double[] getMappedParams() {
    double[] r = Arrays.copyOf(params, params.length);
    for (int i = 0; i < params.length; i++) {
      r[i] = get(i);
    }
    return r;
  }

  public double getOmni() {
    return omni;
  }

  public double[] getStep() {
    return step;
  }

  public T getPayload() {
    return payload;
  }

  public double getValue() {
    return value;
  }

  public void setOmni(double omni) {
    this.omni = omni;
  }

  public void setId(int id) {
    this.id = id;
  }

  public void setStep(double[] step) {
    this.step = step;
  }

  public void setMaps(Mapping[] maps) {
    this.maps = maps;
  }

  public void setMaps(Iterable<Mapping> maps) {
    Collection<Mapping> list = Lists.newArrayList(maps);
    this.maps = list.toArray(new Mapping[list.size()]);
  }

  public void setValue(double v) {
    value = v;
  }

  public void setPayload(T payload) {
    this.payload = payload;
  }

  @Override
  public boolean equals(Object o) {
    if (!(o instanceof State)) {
      return false;
    }
    State<?,?> other = (State<?,?>) o;
    return id == other.id && value == other.value;
  }

  @Override
  public int hashCode() {
    return RandomUtils.hashDouble(value) ^ id;
  }

  /**
   * Natural order is to sort in descending order of score.  Creation order is used as a
   * tie-breaker.
   *
   * @param other The state to compare with.
   * @return -1, 0, 1 if the other state is better, identical or worse than this one.
   */
  @Override
  public int compareTo(State<T, U> other) {
    int r = Double.compare(other.value, this.value);
    if (r != 0) {
      return r;
    }
    if (this.id < other.id) {
      return -1;
    }
    if (this.id > other.id) {
      return 1;
    }
    return 0;
  }

  @Override
  public String toString() {
    double sum = 0;
    for (double v : step) {
      sum += v * v;
    }
    return String.format(Locale.ENGLISH, "<S/%s %.3f %.3f>", payload, omni + Math.sqrt(sum), value);
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(id);
    out.writeInt(params.length);
    for (double v : params) {
      out.writeDouble(v);
    }
    for (Mapping map : maps) {
      PolymorphicWritable.write(out, map);
    }

    out.writeDouble(omni);
    for (double v : step) {
      out.writeDouble(v);
    }

    out.writeDouble(value);
    PolymorphicWritable.write(out, payload);
  }

  @Override
  public void readFields(DataInput input) throws IOException {
    id = input.readInt();
    int n = input.readInt();
    params = new double[n];
    for (int i = 0; i < n; i++) {
      params[i] = input.readDouble();
    }

    maps = new Mapping[n];
    for (int i = 0; i < n; i++) {
      maps[i] = PolymorphicWritable.read(input, Mapping.class);
    }
    omni = input.readDouble();
    step = new double[n];
    for (int i = 0; i < n; i++) {
      step[i] = input.readDouble();
    }
    value = input.readDouble();
    payload = (T) PolymorphicWritable.read(input, Payload.class);
  }
}
