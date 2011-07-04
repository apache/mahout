package org.apache.mahout.math.decomposer.lanczos;

import com.google.common.collect.Maps;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorIterable;

import java.util.Map;

public class LanczosState {
  protected  Matrix diagonalMatrix;
  protected final VectorIterable corpus;
  protected double scaleFactor;
  protected int iterationNumber;
  protected final int desiredRank;
  protected Map<Integer, Vector> basis;

  protected final Map<Integer, Double> singularValues;
  protected Map<Integer, Vector> singularVectors;

  public LanczosState(VectorIterable corpus, int numCols, int desiredRank, Vector initialVector) {
    this.corpus = corpus;
    this.desiredRank = desiredRank;
    intitializeBasisAndSingularVectors(numCols, desiredRank);
    setBasisVector(0, initialVector);
    scaleFactor = 0;
    diagonalMatrix = new DenseMatrix(desiredRank, desiredRank);
    singularValues = Maps.newHashMap();
    iterationNumber = 1;
  }

  protected void intitializeBasisAndSingularVectors(int numCols, int rank) {
    basis = Maps.newHashMap();
    singularVectors = Maps.newHashMap();
  }

  public Matrix getDiagonalMatrix() {
    return diagonalMatrix;
  }

  public int getIterationNumber() {
    return iterationNumber;
  }

  public double getScaleFactor() {
    return scaleFactor;
  }

  public VectorIterable getCorpus() {
    return corpus;
  }

  public Vector getRightSingularVector(int i) {
    return singularVectors.get(i);
  }

  public Double getSingularValue(int i) {
    return singularValues.get(i);
  }

  public Vector getBasisVector(int i) {
    return basis.get(i);
  }

  public void setBasisVector(int i, Vector basisVector) {
    basis.put(i, basisVector);
  }

  public void setScaleFactor(double scale) {
    scaleFactor = scale;
  }

  public void setIterationNumber(int i) {
    iterationNumber = i;
  }

  public void setRightSingularVector(int i, Vector vector) {
    singularVectors.put(i, vector);
  }

  public void setSingularValue(int i, double value) {
    singularValues.put(i, value);
  }
}
