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

package org.apache.mahout.math.hadoop;

import java.io.IOException;
import java.util.Iterator;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix.MatrixEntryWritable;
import org.easymock.IArgumentMatcher;
import org.easymock.EasyMock;

/**
 * a collection of small helper methods useful for unit-testing mathematical operations
 */
public final class MathHelper {

  private MathHelper() {
  }

  /**
   * applies an {@link IArgumentMatcher} to {@link MatrixEntryWritable}s
   */
  public static MatrixEntryWritable matrixEntryMatches(final int row, final int col, final double value) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof MatrixEntryWritable) {
          MatrixEntryWritable entry = (MatrixEntryWritable) argument;
          return row == entry.getRow()
              && col == entry.getCol()
              && Math.abs(value - entry.getVal()) <= MahoutTestCase.EPSILON;
        }
        return false;
      }

      @Override
      public void appendTo(StringBuffer buffer) {
        buffer.append("MatrixEntry[row=").append(row)
            .append(",col=").append(col)
            .append(",value=").append(value).append(']');
      }
    });
    return null;
  }

  /**
   * convenience method to create a {@link MatrixEntryWritable}
   */
  public static MatrixEntryWritable matrixEntry(int row, int col, double value) {
    MatrixEntryWritable entry = new MatrixEntryWritable();
    entry.setRow(row);
    entry.setCol(col);
    entry.setVal(value);
    return entry;
  }

  /**
   * convenience method to create a {@link Vector.Element}
   */
  public static Vector.Element elem(int index, double value) {
    return new ElementToCheck(index, value);
  }

  /**
   * a simple implementation of {@link Vector.Element}
   */
  static class ElementToCheck implements Vector.Element {
    private final int index;
    private double value;

    ElementToCheck(int index, double value) {
      this.index = index;
      this.value = value;
    }
    @Override
    public double get() {
      return value;
    }
    @Override
    public int index() {
      return index;
    }
    @Override
    public void set(double value) {
      this.value = value;
    }
  }

  /**
   * applies an {@link IArgumentMatcher} to a {@link VectorWritable} that checks whether all elements are included
   */
  public static VectorWritable vectorMatches(final Vector.Element... elements) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof VectorWritable) {
          Vector v = ((VectorWritable) argument).get();
          return consistsOf(v, elements);
        }
        return false;
      }

      @Override
      public void appendTo(StringBuffer buffer) {}
    });
    return null;
  }

  /**
   * checks whether the {@link Vector} is equivalent to the set of {@link Vector.Element}s
   */
  public static boolean consistsOf(Vector vector, Vector.Element... elements) {
    if (elements.length != numberOfNoNZeroNonNaNElements(vector)) {
      return false;
    }
    for (Vector.Element element : elements) {
      boolean matches = Math.abs(element.get() - vector.get(element.index())) <= MahoutTestCase.EPSILON;
      if (!matches) {
        return false;
      }
    }
    return true;    
  }
  
  /**
   * returns the number of elements in the {@link Vector} that are neither 0 nor NaN
   */
  public static int numberOfNoNZeroNonNaNElements(Vector vector) {
    int elementsInVector = 0;
    Iterator<Vector.Element> vectorIterator = vector.iterateNonZero();
    while (vectorIterator.hasNext()) {
      Vector.Element currentElement = vectorIterator.next();
      if (!Double.isNaN(currentElement.get())) {
        elementsInVector++;
      }      
    }
    return elementsInVector;
  }
  
  /**
   * read a {@link Matrix} from a SequenceFile<IntWritable,VectorWritable>
   */
  public static Matrix readEntries(Configuration conf, Path path, int rows, int columns) {
    Matrix matrix = new DenseMatrix(rows, columns);
    for (Pair<IntWritable,VectorWritable> record :
         new SequenceFileIterable<IntWritable,VectorWritable>(path, true, conf)) {
      IntWritable key = record.getFirst();
      VectorWritable value = record.getSecond();
      int row = key.get();
      Iterator<Vector.Element> elementsIterator = value.get().iterateNonZero();
      while (elementsIterator.hasNext()) {
        Vector.Element element = elementsIterator.next();
        matrix.set(row, element.index(), element.get());
      }
    }
    return matrix;
  }

  /**
   * write a two-dimensional double array to an SequenceFile<IntWritable,VectorWritable>
   */
  public static void writeEntries(double[][] entries, FileSystem fs, Configuration conf, Path path)
      throws IOException {
    SequenceFile.Writer writer = null;
    try {
      writer = new SequenceFile.Writer(fs, conf, path, IntWritable.class, VectorWritable.class);
      for (int n = 0; n < entries.length; n++) {
        Vector v = new RandomAccessSparseVector(entries[n].length);
        for (int m = 0; m < entries[n].length; m++) {
          v.setQuick(m, entries[n][m]);
        }
        writer.append(new IntWritable(n), new VectorWritable(v));
      }
    } finally {
      Closeables.closeQuietly(writer);
    }
  }
}
