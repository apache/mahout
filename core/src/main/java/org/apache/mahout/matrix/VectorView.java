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

package org.apache.mahout.matrix;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Iterator;
import java.nio.charset.Charset;

/**
 * Implements subset view of a Vector
 */
public class VectorView extends AbstractVector {

  /** For serialization purposes only */
  public VectorView() {
  }

  private Vector vector;

  // the offset into the Vector
  private int offset;

  // the cardinality of the view
  private int cardinality;

  public VectorView(Vector vector, int offset, int cardinality) {
    super();
    this.vector = vector;
    this.offset = offset;
    this.cardinality = cardinality;
  }

  @Override
  protected Matrix matrixLike(int rows, int columns) {
    return ((AbstractVector) vector).matrixLike(rows, columns);
  }

  @Override
  public WritableComparable asWritableComparable() {
    String out = asFormatString();
    return new Text(out);
  }

  @Override
  public String asFormatString() {
    StringBuilder out = new StringBuilder();
    out.append('[');
    for (int i = offset; i < offset + cardinality; i++)
      out.append(getQuick(i)).append(", ");
    out.append("] ");
    return out.toString();
  }

  @Override
  public int cardinality() {
    return cardinality;
  }

  @Override
  public Vector copy() {
    return new VectorView(vector.copy(), offset, cardinality);
  }

  @Override
  public double getQuick(int index) {
    return vector.getQuick(offset + index);
  }

  @Override
  public Vector like() {
    return vector.like();
  }

  @Override
  public Vector like(int cardinality) {
    return vector.like(cardinality);
  }

  @Override
  public void setQuick(int index, double value) {
    vector.setQuick(offset + index, value);
  }

  @Override
  public int size() {
    return cardinality;
  }

  @Override
  public double[] toArray() {
    double[] result = new double[cardinality];
    for (int i = 0; i < cardinality; i++)
      result[i] = vector.getQuick(offset + i);
    return result;
  }

  @Override
  public Vector viewPart(int offset, int length) {
    if (length > cardinality)
      throw new CardinalityException();
    if (offset < 0 || offset + length > cardinality)
      throw new IndexException();
    return new VectorView(vector, offset + this.offset, length);
  }

  @Override
  public boolean haveSharedCells(Vector other) {
    if (other instanceof VectorView)
      return other == this || vector.haveSharedCells(other);
    else
      return other.haveSharedCells(vector);
  }

  /**
   * @return true if index is a valid index in the underlying Vector
   */
  private boolean isInView(int index) {
    return index >= offset && index < offset + cardinality;
  }

  @Override
  public Iterator<Vector.Element> iterator() {
    return new ViewIterator();
  }

  public class ViewIterator implements Iterator<Vector.Element> {
    private final Iterator<Vector.Element> it;
    private Vector.Element el;

    public ViewIterator() {
      it = vector.iterator();
      buffer();
    }

    private void buffer() {
      while (it.hasNext()) {
        el = it.next();
        if (isInView(el.index())) {
          final Vector.Element decorated = el;
          el = new Vector.Element() {
            public double get() {
              return decorated.get();
            }

            public int index() {
              return decorated.index() - offset;
            }

            public void set(double value) {
              el.set(value);
            }
          };
          return;
        }
      }
      el = null; // No element was found
    }

    public Vector.Element next() {
      Vector.Element buffer = el;
      buffer();
      return buffer;
    }

    public boolean hasNext() {
      return el != null;
    }

    /**
     * @throws UnsupportedOperationException
     *             all the time. method not implemented.
     */
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }


  public void write(DataOutput dataOutput) throws IOException {
    dataOutput.writeInt(offset);
    dataOutput.writeInt(cardinality);
    String vectorClassName = vector.getClass().getName();
    dataOutput.writeInt(vectorClassName.length() * 2);
    dataOutput.write(vectorClassName.getBytes());
    vector.write(dataOutput);
  }

  public void readFields(DataInput dataInput) throws IOException {
    int offset = dataInput.readInt();
    int cardinality = dataInput.readInt();
    byte[] buf = new byte[dataInput.readInt()];
    dataInput.readFully(buf);
    String vectorClassName = new String(buf, Charset.forName("UTF-8"));
    Vector vector;
    try {
      vector = Class.forName(vectorClassName).asSubclass(Vector.class).newInstance();
    } catch (ClassNotFoundException e) {
      throw new RuntimeException(e);
    } catch (IllegalAccessException e) {
      throw new RuntimeException(e);
    } catch (InstantiationException e) {
      throw new RuntimeException(e);
    }
    vector.readFields(dataInput);

    this.offset = offset;
    this.cardinality = cardinality;
    this.vector = vector;
  }
}
