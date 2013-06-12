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
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;

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
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.map.OpenIntObjectHashMap;
import org.easymock.EasyMock;
import org.easymock.IArgumentMatcher;
import org.junit.Assert;

/**
 * a collection of small helper methods useful for unit-testing mathematical operations
 */
public final class MathHelper {

  private MathHelper() {}

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
      if (Math.abs(element.get() - vector.get(element.index())) > MahoutTestCase.EPSILON) {
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
    for (Element currentElement : vector.nonZeroes()) {
      if (!Double.isNaN(currentElement.get())) {
        elementsInVector++;
      }      
    }
    return elementsInVector;
  }
  
  /**
   * read a {@link Matrix} from a SequenceFile<IntWritable,VectorWritable>
   */
  public static Matrix readMatrix(Configuration conf, Path path, int rows, int columns) {
    boolean readOneRow = false;
    Matrix matrix = new DenseMatrix(rows, columns);
    for (Pair<IntWritable,VectorWritable> record :
        new SequenceFileIterable<IntWritable,VectorWritable>(path, true, conf)) {
      IntWritable key = record.getFirst();
      VectorWritable value = record.getSecond();
      readOneRow = true;
      int row = key.get();
      for (Element element : value.get().nonZeroes()) {
        matrix.set(row, element.index(), element.get());
      }
    }
    if (!readOneRow) {
      throw new IllegalStateException("Not a single row read!");
    }
    return matrix;
  }

  /**
   * read a {@link Matrix} from a SequenceFile<IntWritable,VectorWritable>
   */
  public static OpenIntObjectHashMap<Vector> readMatrixRows(Configuration conf, Path path) {
    boolean readOneRow = false;
    OpenIntObjectHashMap<Vector> rows = new OpenIntObjectHashMap<Vector>();
    for (Pair<IntWritable,VectorWritable> record :
        new SequenceFileIterable<IntWritable,VectorWritable>(path, true, conf)) {
      IntWritable key = record.getFirst();
      readOneRow = true;
      rows.put(key.get(), record.getSecond().get());
    }
    if (!readOneRow) {
      throw new IllegalStateException("Not a single row read!");
    }
    return rows;
  }

  /**
   * write a two-dimensional double array to an SequenceFile<IntWritable,VectorWritable>
   */
  public static void writeDistributedRowMatrix(double[][] entries, FileSystem fs, Configuration conf, Path path)
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
      Closeables.close(writer, false);
    }
  }

  public static void assertMatrixEquals(Matrix expected, Matrix actual) {
    Assert.assertEquals(expected.numRows(), actual.numRows());
    Assert.assertEquals(actual.numCols(), actual.numCols());
    for (int row = 0; row < expected.numRows(); row++) {
      for (int col = 0; col < expected.numCols(); col ++) {
        Assert.assertEquals("Non-matching values in [" + row + ',' + col + ']',
                            expected.get(row, col), actual.get(row, col), MahoutTestCase.EPSILON);
      }
    }
  }

  public static String nice(Vector v) {
    if (!v.isSequentialAccess()) {
      v = new DenseVector(v);
    }

    DecimalFormat df = new DecimalFormat("0.00", DecimalFormatSymbols.getInstance(Locale.ENGLISH));

    StringBuilder buffer = new StringBuilder("[");
    String separator = "";
    for (Vector.Element e : v.all()) {
      buffer.append(separator);
      if (Double.isNaN(e.get())) {
        buffer.append("  -  ");
      } else {
        if (e.get() >= 0) {
          buffer.append(' ');
        }
        buffer.append(df.format(e.get()));
      }
      separator = "\t";
    }
    buffer.append(" ]");
    return buffer.toString();
  }

  public static String nice(Matrix matrix) {
    StringBuilder info = new StringBuilder();
    for (int n = 0; n < matrix.numRows(); n++) {
      info.append(nice(matrix.viewRow(n))).append('\n');
    }
    return info.toString();
  }
}
