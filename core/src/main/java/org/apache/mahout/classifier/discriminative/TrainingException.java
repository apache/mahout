package org.apache.mahout.classifier.discriminative;

/**
 * This exception is thrown in case training fails. E.g. training with an algorithm
 * that can find linear separating hyperplanes only on a training set that is not
 * linearly separable.
 * */
public class TrainingException extends Exception {
  /** Serialization id. */
  private static final long serialVersionUID = 388611231310145397L;

  /**
   * Init with message string describing the cause of the exception.
   * */
  public TrainingException(String message) {
    super(message);
  }
}
