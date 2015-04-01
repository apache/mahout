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

package org.apache.mahout.math.hadoop.stochasticsvd;

import com.google.common.base.Function;
import com.google.common.collect.Iterators;
import com.google.common.io.Closeables;
import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.IOUtils;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterator;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseSymmetricMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.UpperTriangular;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

/**
 * set of small file manipulation helpers.
 */
public final class SSVDHelper {

  private static final Pattern OUTPUT_FILE_PATTERN = Pattern.compile("(\\w+)-(m|r)-(\\d+)(\\.\\w+)?");

  private SSVDHelper() {
  }

  /**
   * load single vector from an hdfs file (possibly presented as glob).
   */
  static Vector loadVector(Path glob, Configuration conf) throws IOException {

    SequenceFileDirValueIterator<VectorWritable> iter =
      new SequenceFileDirValueIterator<>(glob,
                                                       PathType.GLOB,
                                                       null,
                                                       null,
                                                       true,
                                                       conf);

    try {
      if (!iter.hasNext()) {
        throw new IOException("Empty input while reading vector");
      }
      VectorWritable vw = iter.next();

      if (iter.hasNext()) {
        throw new IOException("Unexpected data after the end of vector file");
      }

      return vw.get();

    } finally {
      Closeables.close(iter, true);
    }
  }

  /**
   * save single vector into hdfs file.
   *
   * @param v vector to save
   */
  public static void saveVector(Vector v,
                                Path vectorFilePath,
                                Configuration conf) throws IOException {
    VectorWritable vw = new VectorWritable(v);
    FileSystem fs = FileSystem.get(conf);
    try (SequenceFile.Writer w = new SequenceFile.Writer(fs,
        conf,
        vectorFilePath,
        IntWritable.class,
        VectorWritable.class)) {
      w.append(new IntWritable(), vw);
    }
      /*
       * this is a writer, no quiet close please. we must bail out on incomplete
       * close.
       */

  }

  /**
   * sniff label type in the input files
   */
  static Class<? extends Writable> sniffInputLabelType(Path[] inputPath,
                                                       Configuration conf)
    throws IOException {
    FileSystem fs = FileSystem.get(conf);
    for (Path p : inputPath) {
      FileStatus[] fstats = fs.globStatus(p);
      if (fstats == null || fstats.length == 0) {
        continue;
      }

      FileStatus firstSeqFile;
      if (fstats[0].isDir()) {
        firstSeqFile = fs.listStatus(fstats[0].getPath(), PathFilters.logsCRCFilter())[0];
      } else {
        firstSeqFile = fstats[0];
      }

      SequenceFile.Reader r = null;
      try {
        r = new SequenceFile.Reader(fs, firstSeqFile.getPath(), conf);
        return r.getKeyClass().asSubclass(Writable.class);
      } finally {
        Closeables.close(r, true);
      }
    }
    throw new IOException("Unable to open input files to determine input label type.");
  }

  static final Comparator<FileStatus> PARTITION_COMPARATOR =
    new Comparator<FileStatus>() {
      private final Matcher matcher = OUTPUT_FILE_PATTERN.matcher("");

      @Override
      public int compare(FileStatus o1, FileStatus o2) {
        matcher.reset(o1.getPath().getName());
        if (!matcher.matches()) {
          throw new IllegalArgumentException("Unexpected file name, unable to deduce partition #:"
                                               + o1.getPath());
        }
        int p1 = Integer.parseInt(matcher.group(3));
        matcher.reset(o2.getPath().getName());
        if (!matcher.matches()) {
          throw new IllegalArgumentException("Unexpected file name, unable to deduce partition #:"
                                               + o2.getPath());
        }

        int p2 = Integer.parseInt(matcher.group(3));
        return p1 - p2;
      }

    };

  public static Iterator<Pair<Writable, Vector>> drmIterator(FileSystem fs, Path glob, Configuration conf,
                                                             Deque<Closeable> closeables)
    throws IOException {
    SequenceFileDirIterator<Writable, VectorWritable> ret =
      new SequenceFileDirIterator<>(glob,
                                                            PathType.GLOB,
                                                            PathFilters.logsCRCFilter(),
                                                            PARTITION_COMPARATOR,
                                                            true,
                                                            conf);
    closeables.addFirst(ret);
    return Iterators.transform(ret, new Function<Pair<Writable, VectorWritable>, Pair<Writable, Vector>>() {
      @Override
      public Pair<Writable, Vector> apply(Pair<Writable, VectorWritable> p) {
        return new Pair(p.getFirst(), p.getSecond().get());
      }
    });
  }

  /**
   * helper capabiltiy to load distributed row matrices into dense matrix (to
   * support tests mainly).
   *
   * @param fs   filesystem
   * @param glob FS glob
   * @param conf configuration
   * @return Dense matrix array
   */
  public static DenseMatrix drmLoadAsDense(FileSystem fs, Path glob, Configuration conf) throws IOException {

    Deque<Closeable> closeables = new ArrayDeque<>();
    try {
      List<double[]> denseData = new ArrayList<>();
      for (Iterator<Pair<Writable, Vector>> iter = drmIterator(fs, glob, conf, closeables);
           iter.hasNext(); ) {
        Pair<Writable, Vector> p = iter.next();
        Vector v = p.getSecond();
        double[] dd = new double[v.size()];
        if (v.isDense()) {
          for (int i = 0; i < v.size(); i++) {
            dd[i] = v.getQuick(i);
          }
        } else {
          for (Vector.Element el : v.nonZeroes()) {
            dd[el.index()] = el.get();
          }
        }
        denseData.add(dd);
      }
      if (denseData.size() == 0) {
        return null;
      } else {
        return new DenseMatrix(denseData.toArray(new double[denseData.size()][]));
      }
    } finally {
      IOUtils.close(closeables);
    }
  }

  /**
   * Load multiple upper triangular matrices and sum them up.
   *
   * @return the sum of upper triangular inputs.
   */
  public static DenseSymmetricMatrix loadAndSumUpperTriangularMatricesAsSymmetric(Path glob, Configuration conf) throws IOException {
    Vector v = loadAndSumUpVectors(glob, conf);
    return v == null ? null : new DenseSymmetricMatrix(v);
  }

  /**
   * @return sum of all vectors in different files specified by glob
   */
  public static Vector loadAndSumUpVectors(Path glob, Configuration conf)
    throws IOException {

    SequenceFileDirValueIterator<VectorWritable> iter =
      new SequenceFileDirValueIterator<>(glob,
                                                       PathType.GLOB,
                                                       null,
                                                       PARTITION_COMPARATOR,
                                                       true,
                                                       conf);

    try {
      Vector v = null;
      while (iter.hasNext()) {
        if (v == null) {
          v = new DenseVector(iter.next().get());
        } else {
          v.assign(iter.next().get(), Functions.PLUS);
        }
      }
      return v;

    } finally {
      Closeables.close(iter, true);
    }

  }

  /**
   * Load only one upper triangular matrix and issue error if mroe than one is
   * found.
   */
  public static UpperTriangular loadUpperTriangularMatrix(Path glob, Configuration conf) throws IOException {

    /*
     * there still may be more than one file in glob and only one of them must
     * contain the matrix.
     */

    try (SequenceFileDirValueIterator<VectorWritable> iter = new SequenceFileDirValueIterator<>(glob,
        PathType.GLOB,
        null,
        null,
        true,
        conf)) {
      if (!iter.hasNext()) {
        throw new IOException("No triangular matrices found");
      }
      Vector v = iter.next().get();
      UpperTriangular result = new UpperTriangular(v);
      if (iter.hasNext()) {
        throw new IOException("Unexpected overrun in upper triangular matrix files");
      }
      return result;

    }
  }

  /**
   * extracts row-wise raw data from a Mahout matrix for 3rd party solvers.
   * Unfortunately values member is 100% encapsulated in {@link org.apache.mahout.math.DenseMatrix} at
   * this point, so we have to resort to abstract element-wise copying.
   */
  public static double[][] extractRawData(Matrix m) {
    int rows = m.numRows();
    int cols = m.numCols();
    double[][] result = new double[rows][];
    for (int i = 0; i < rows; i++) {
      result[i] = new double[cols];
      for (int j = 0; j < cols; j++) {
        result[i][j] = m.getQuick(i, j);
      }
    }
    return result;
  }

}
