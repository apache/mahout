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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.common.IOUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.hadoop.DistributedRowMatrix.MatrixEntryWritable;
import org.easymock.IArgumentMatcher;
import org.easymock.classextension.EasyMock;

/**
 * a collection of small helper methods useful for unit-testing mathematical operations
 */
public class MathHelper {

  /** the "close enough" value for floating point computations */
  public static final double EPSILON = 0.00001d;

  private MathHelper() {
  }

  /**
   * applies an {@link IArgumentMatcher} to {@link MatrixEntryWritable}s
   *
   * @param row
   * @param col
   * @param value
   * @return
   */
  public static MatrixEntryWritable matrixEntryMatches(final int row, final int col, final double value) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof MatrixEntryWritable) {
          MatrixEntryWritable entry = (MatrixEntryWritable) argument;
          return (row == entry.getRow() && col == entry.getCol() && Math.abs(value - entry.getVal()) <= EPSILON);
        }
        return false;
      }

      @Override
      public void appendTo(StringBuffer buffer) {}
    });
    return null;
  }

  /**
   * convenience method to create a {@link MatrixEntryWritable}
   *
   * @param row
   * @param col
   * @param value
   * @return
   */
  public static MatrixEntryWritable matrixEntry(int row, int col, double value) {
    MatrixEntryWritable entry = new MatrixEntryWritable();
    entry.setRow(row);
    entry.setCol(col);
    entry.setVal(value);
    return entry;
  }

  /**
   * convenience method to create a {@link Element}
   *
   * @param index
   * @param value
   * @return
   */
  public static Element elem(int index, double value) {
    return new ElementToCheck(index, value);
  }

  /**
   * a simple implementation of {@link Element}
   */
  static class ElementToCheck implements Element {
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
   *
   * @param elements
   * @return
   */
  public static VectorWritable vectorMatches(final Element... elements) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof VectorWritable) {
          Vector v = ((VectorWritable) argument).get();
          for (Element element : elements) {
            boolean matches = Math.abs(element.get() - v.get(element.index())) <= EPSILON;
            if (!matches) {
              return false;
            }
          }
          return true;
        }
        return false;
      }

      @Override
      public void appendTo(StringBuffer buffer) {}
    });
    return null;
  }

  /**
   * read a {@link Matrix} from a SequenceFile<IntWritable,VectorWritable>
   * @param fs
   * @param conf
   * @param path
   * @param rows
   * @param columns
   * @return
   * @throws IOException
   */
  public static Matrix readEntries(FileSystem fs, Configuration conf, Path path, int rows, int columns)
      throws IOException {

    Matrix matrix = new DenseMatrix(rows, columns);

    SequenceFile.Reader reader = null;
    try {
      reader = new SequenceFile.Reader(fs, path, conf);
      IntWritable key = new IntWritable();
      VectorWritable value = new VectorWritable();
      while (reader.next(key, value)) {
        int row = key.get();
        Iterator<Element> elementsIterator = value.get().iterateNonZero();
        while (elementsIterator.hasNext()) {
          Element element = elementsIterator.next();
          matrix.set(row, element.index(), element.get());
        }
      }
    } finally {
      IOUtils.quietClose(reader);
    }
    return matrix;
  }

  /**
   * write a two-dimensional double array to an SequenceFile<IntWritable,VectorWritable>
   *
   * @param entries
   * @param fs
   * @param conf
   * @param path
   * @throws IOException
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
      IOUtils.quietClose(writer);
    }
  }
}
