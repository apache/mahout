package org.apache.mahout.utils.vectors;

import org.apache.mahout.matrix.Vector;
import org.apache.mahout.matrix.DenseVector;
import org.apache.mahout.matrix.UnaryFunction;
import org.apache.mahout.matrix.SparseVector;

import java.util.Iterator;
import java.util.Random;


/**
 *
 *
 **/
public class RandomVectorIterable implements VectorIterable{

  int numItems = 100;
  public static enum VectorType {DENSE, SPARSE};

  VectorType type = VectorType.SPARSE;

  public RandomVectorIterable() {
  }

  public RandomVectorIterable(int numItems) {
    this.numItems = numItems;
  }

  public RandomVectorIterable(int numItems, VectorType type) {
    this.numItems = numItems;
    this.type = type;
  }

  @Override
  public Iterator<Vector> iterator() {
    return new VectIterator();
  }

  private class VectIterator implements Iterator<Vector>{
    int count = 0;
    Random random = new Random();
    @Override
    public boolean hasNext() {
      return count < numItems;
    }

    @Override
    public Vector next() {
      Vector result = type.equals(VectorType.SPARSE) ? new SparseVector(numItems) : new DenseVector(numItems);
      result.assign(new UnaryFunction(){
        @Override
        public double apply(double arg1) {
          return random.nextDouble();
        }
      });
      count++;
      return result;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }
}
