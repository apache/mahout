package org.apache.mahout.math;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ReflectionUtils;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class VectorWritable extends Configured implements Writable {

  private Vector vector;
  // cache most recent vector instance class name
  private static String instanceClassName;// cache most recent vector instance class
  private static Class<? extends Vector> instanceClass;

  public Vector get() {
    return vector;
  }

  public void set(Vector vector) {
    this.vector = vector;
  }

  public VectorWritable() {

  }

  public VectorWritable(Vector v) {
    set(v);
  }

  @Override
  public void write(DataOutput out) throws IOException {
    Writable w = null;
    if(vector instanceof Writable) {
      w = (Writable) vector;
    } else if(vector instanceof SparseVector) {
      w = new SparseVectorWritable((SparseVector)vector);
    } else {
      w = new DenseVectorWritable(new DenseVector(vector));
    }
    w.write(out);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    try {
      String vectorClassName = in.readUTF();
      Class<? extends Vector> inputClass = (Class<? extends Vector>) getConf().getClassByName(vectorClassName);
      Class<? extends Vector> vectorClass = getConf().getClass("vector.class", inputClass, Vector.class);
      vector = ReflectionUtils.newInstance(vectorClass, getConf());
      ((Writable)vector).readFields(in);
    } catch (ClassNotFoundException cnfe) {
      throw new IOException(cnfe);
    } catch (ClassCastException cce) {
      throw new IOException(cce);
    }
  }

  /** Read and return a vector from the input */
  public static Vector readVector(DataInput in) throws IOException {
    String vectorClassName = in.readUTF();
    Vector vector;
    try {
      if (!vectorClassName.equals(instanceClassName)) {
        instanceClassName = vectorClassName;
        instanceClass = Class.forName(vectorClassName).asSubclass(Vector.class);
      }
      vector = instanceClass.newInstance();
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    }
    ((Writable)vector).readFields(in);
    return vector;
  }

  /** Write the vector to the output */
  public static void writeVector(DataOutput out, Vector vector)
      throws IOException {
    new VectorWritable(vector).write(out);
  }
}
