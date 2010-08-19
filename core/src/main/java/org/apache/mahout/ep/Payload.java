package org.apache.mahout.ep;

/**
 * Payloads for evolutionary state must be copyable and updatable.  The copy should be
 * a deep copy unless some aspect of the state is sharable or immutable.
 *
 * During mutation, a copy is first made and then after the parameters in the State
 * structure are suitably modified, update is called with the scaled versions of the
 * parameters.
 *
 * @see State
 * @param <T>
 */
public interface Payload<T> {
  public T copy();
  public void update(double[] params);
}
