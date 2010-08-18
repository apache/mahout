package org.apache.mahout.ep;

import java.util.Arrays;
import java.util.Random;

/**
 * Recorded step evolutionary optimization.  You provide the value, this class provides the
 * mutation.
 */
public class State implements Comparable<State> {
  // object count is kept to break ties in comparison.
  static volatile int objectCount = 0;
  private Random gen = new Random();

  int id = objectCount++;

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
   * Deep clones a state, useful in mutation.
   *
   * @param params Current state
   * @param omni   Current omni-directional mutation
   * @param step   The step taken to get to this point
   */
  private State(double[] params, double omni, double[] step, Mapping[] maps) {
    this.params = Arrays.copyOf(params, params.length);
    this.omni = omni;
    this.step = Arrays.copyOf(step, step.length);
    this.maps = Arrays.copyOf(maps, maps.length);
  }

  /**
   * Clone this state with a random change in position.
   *
   * @return A new state.
   */
  public State mutate() {
    double sum = 0;
    for (double v : step) {
      sum += v * v;
    }
    sum = Math.sqrt(sum);
    double lambda = 0.9 + gen.nextGaussian();
    State r = new State(params, omni, step, maps);
    r.omni = -Math.log(1 - gen.nextDouble()) * (0.9 * omni + sum / 10);
    for (int i = 0; i < step.length; i++) {
      r.step[i] = lambda * step[i] + r.omni * gen.nextGaussian();
      r.params[i] += r.step[i];
    }
    return r;
  }

  /**
   * Defines the transformation for a parameter.
   * @param i Which mapping to define.
   * @param m The mapping to use.
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
    if (m == null) {
      return params[i];
    } else {
      return m.apply(params[i]);
    }
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

  public double[] getParams() {
    return params;
  }

  public double getOmni() {
    return omni;
  }

  public double[] getStep() {
    return step;
  }


  /**
   * Natural order is to sort in descending order of score.  Creation order is used as a
   * tie-breaker.
   *
   * @param other The state to compare with.
   * @return -1, 0, 1 if the other state is better, identical or worse than this one.
   */
  @Override
  public int compareTo(State other) {
    int r = Double.compare(other.value, this.value);
    if (r != 0) {
      return r;
    } else {
      return this.id - other.id;
    }
  }

  public void setRand(Random rand) {
    this.gen = rand;
  }

  public void setOmni(double omni) {
    this.omni = omni;
  }

  public void setValue(double v) {
    value = v;
  }
}
