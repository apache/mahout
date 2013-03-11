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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterator;
import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorIterable;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Function;
import com.google.common.collect.Iterators;

/**
 * DistributedRowMatrix is a FileSystem-backed VectorIterable in which the vectors live in a
 * SequenceFile<WritableComparable,VectorWritable>, and distributed operations are executed as M/R passes on
 * Hadoop.  The usage is as follows: <p>
 * <p>
 * <pre>
 *   // the path must already contain an already created SequenceFile!
 *   DistributedRowMatrix m = new DistributedRowMatrix("path/to/vector/sequenceFile", "tmp/path", 10000000, 250000);
 *   m.setConf(new Configuration());
 *   // now if we want to multiply a vector by this matrix, it's dimension must equal the row dimension of this
 *   // matrix.  If we want to timesSquared() a vector by this matrix, its dimension must equal the column dimension
 *   // of the matrix.
 *   Vector v = new DenseVector(250000);
 *   // now the following operation will be done via a M/R pass via Hadoop.
 *   Vector w = m.timesSquared(v);
 * </pre>
 *
 */
public class DistributedRowMatrix implements VectorIterable, Configurable {
  public static final String KEEP_TEMP_FILES = "DistributedMatrix.keep.temp.files";

  private static final Logger log = LoggerFactory.getLogger(DistributedRowMatrix.class);

  private final Path inputPath;
  private final Path outputTmpPath;
  private Configuration conf;
  private Path rowPath;
  private Path outputTmpBasePath;
  private final int numRows;
  private final int numCols;
  private boolean keepTempFiles;

  public DistributedRowMatrix(Path inputPath,
                              Path outputTmpPath,
                              int numRows,
                              int numCols) {
    this(inputPath, outputTmpPath, numRows, numCols, false);
  }

  public DistributedRowMatrix(Path inputPath,
                              Path outputTmpPath,
                              int numRows,
                              int numCols,
                              boolean keepTempFiles) {
    this.inputPath = inputPath;
    this.outputTmpPath = outputTmpPath;
    this.numRows = numRows;
    this.numCols = numCols;
    this.keepTempFiles = keepTempFiles;
  }

  @Override
  public Configuration getConf() {
    return conf;
  }

  @Override
  public void setConf(Configuration conf) {
    this.conf = conf;
    try {
      FileSystem fs = FileSystem.get(inputPath.toUri(), conf);
      rowPath = fs.makeQualified(inputPath);
      outputTmpBasePath = fs.makeQualified(outputTmpPath);
      keepTempFiles = conf.getBoolean(KEEP_TEMP_FILES, false);
    } catch (IOException ioe) {
      throw new IllegalStateException(ioe);
    }
  }

  public Path getRowPath() {
    return rowPath;
  }

  public Path getOutputTempPath() {
    return outputTmpBasePath;
  }

  public void setOutputTempPathString(String outPathString) {
    try {
      outputTmpBasePath = FileSystem.get(conf).makeQualified(new Path(outPathString));
    } catch (IOException ioe) {
      log.warn("Unable to set outputBasePath to {}, leaving as {}",
          outPathString, outputTmpBasePath);
    }
  }

  @Override
  public Iterator<MatrixSlice> iterateAll() {
    try {
      Path pathPattern = rowPath;
      if (FileSystem.get(conf).getFileStatus(rowPath).isDir()) {
        pathPattern = new Path(rowPath, "*");
      }
      return Iterators.transform(
          new SequenceFileDirIterator<IntWritable,VectorWritable>(pathPattern,
                                                                  PathType.GLOB,
                                                                  PathFilters.logsCRCFilter(),
                                                                  null,
                                                                  true,
                                                                  conf),
          new Function<Pair<IntWritable,VectorWritable>,MatrixSlice>() {
            @Override
            public MatrixSlice apply(Pair<IntWritable, VectorWritable> from) {
              return new MatrixSlice(from.getSecond().get(), from.getFirst().get());
            }
          });
    } catch (IOException ioe) {
      throw new IllegalStateException(ioe);
    }
  }

  @Override
  public int numSlices() {
    return numRows();
  }

  @Override
  public int numRows() {
    return numRows;
  }

  @Override
  public int numCols() {
    return numCols;
  }


  /**
   * This implements matrix this.transpose().times(other)
   * @param other   a DistributedRowMatrix
   * @return    a DistributedRowMatrix containing the product
   */
  public DistributedRowMatrix times(DistributedRowMatrix other) throws IOException {
    return times(other, new Path(outputTmpBasePath.getParent(), "productWith-" + (System.nanoTime() & 0xFF)));
  }

  /**
   * This implements matrix this.transpose().times(other)
   * @param other   a DistributedRowMatrix
   * @param outPath path to write result to
   * @return    a DistributedRowMatrix containing the product
   */
  public DistributedRowMatrix times(DistributedRowMatrix other, Path outPath) throws IOException {
    if (numRows != other.numRows()) {
      throw new CardinalityException(numRows, other.numRows());
    }

    Configuration initialConf = getConf() == null ? new Configuration() : getConf();
    Configuration conf =
        MatrixMultiplicationJob.createMatrixMultiplyJobConf(initialConf,
                                                            rowPath,
                                                            other.rowPath,
                                                            outPath,
                                                            other.numCols);
    JobClient.runJob(new JobConf(conf));
    DistributedRowMatrix out = new DistributedRowMatrix(outPath, outputTmpPath, numCols, other.numCols());
    out.setConf(conf);
    return out;
  }

  public Vector columnMeans() throws IOException {
    return columnMeans("SequentialAccessSparseVector");
  }

  /**
   * Returns the column-wise mean of a DistributedRowMatrix
   *
   * @param vectorClass
   *          desired class for the column-wise mean vector e.g.
   *          RandomAccessSparseVector, DenseVector
   * @return Vector containing the column-wise mean of this
   */
  public Vector columnMeans(String vectorClass) throws IOException {
    Path outputVectorTmpPath =
        new Path(outputTmpBasePath, new Path(Long.toString(System.nanoTime())));
    Configuration initialConf =
        getConf() == null ? new Configuration() : getConf();
    String vectorClassFull = "org.apache.mahout.math." + vectorClass;
    Vector mean = MatrixColumnMeansJob.run(initialConf, rowPath, outputVectorTmpPath, vectorClassFull);
    if (!keepTempFiles) {
      FileSystem fs = outputVectorTmpPath.getFileSystem(conf);
      fs.delete(outputVectorTmpPath, true);
    }
    return mean;
  }

  public DistributedRowMatrix transpose() throws IOException {
    Path outputPath = new Path(rowPath.getParent(), "transpose-" + (System.nanoTime() & 0xFF));
    Configuration initialConf = getConf() == null ? new Configuration() : getConf();
    Configuration conf = TransposeJob.buildTransposeJobConf(initialConf, rowPath, outputPath, numRows);
    JobClient.runJob(new JobConf(conf));
    DistributedRowMatrix m = new DistributedRowMatrix(outputPath, outputTmpPath, numCols, numRows);
    m.setConf(this.conf);
    return m;
  }

  @Override
  public Vector times(Vector v) {
    try {
      Configuration initialConf = getConf() == null ? new Configuration() : getConf();
      Path outputVectorTmpPath = new Path(outputTmpBasePath,
                                          new Path(Long.toString(System.nanoTime())));
      Configuration conf =
          TimesSquaredJob.createTimesJobConf(initialConf,
                                             v,
                                             numRows,
                                             rowPath,
                                             outputVectorTmpPath);
      JobClient.runJob(new JobConf(conf));
      Vector result = TimesSquaredJob.retrieveTimesSquaredOutputVector(conf);
      if (!keepTempFiles) {
        FileSystem fs = outputVectorTmpPath.getFileSystem(conf);
        fs.delete(outputVectorTmpPath, true);
      }
      return result;
    } catch (IOException ioe) {
      throw new IllegalStateException(ioe);
    }
  }

  @Override
  public Vector timesSquared(Vector v) {
    try {
      Configuration initialConf = getConf() == null ? new Configuration() : getConf();
      Path outputVectorTmpPath = new Path(outputTmpBasePath,
               new Path(Long.toString(System.nanoTime())));
      Configuration conf =
          TimesSquaredJob.createTimesSquaredJobConf(initialConf,
                                                    v,
                                                    rowPath,
                                                    outputVectorTmpPath);
      JobClient.runJob(new JobConf(conf));
      Vector result = TimesSquaredJob.retrieveTimesSquaredOutputVector(conf);
      if (!keepTempFiles) {
        FileSystem fs = outputVectorTmpPath.getFileSystem(conf);
        fs.delete(outputVectorTmpPath, true);
      }
      return result;
    } catch (IOException ioe) {
      throw new IllegalStateException(ioe);
    }
  }

  @Override
  public Iterator<MatrixSlice> iterator() {
    return iterateAll();
  }

  public static class MatrixEntryWritable implements WritableComparable<MatrixEntryWritable> {
    private int row;
    private int col;
    private double val;

    public int getRow() {
      return row;
    }

    public void setRow(int row) {
      this.row = row;
    }

    public int getCol() {
      return col;
    }

    public void setCol(int col) {
      this.col = col;
    }

    public double getVal() {
      return val;
    }

    public void setVal(double val) {
      this.val = val;
    }

    @Override
    public int compareTo(MatrixEntryWritable o) {
      if (row > o.row) {
        return 1;
      } else if (row < o.row) {
        return -1;
      } else {
        if (col > o.col) {
          return 1;
        } else if (col < o.col) {
          return -1;
        } else {
          return 0;
        }
      }
    }

    @Override
    public boolean equals(Object o) {
      if (!(o instanceof MatrixEntryWritable)) {
        return false;
      }
      MatrixEntryWritable other = (MatrixEntryWritable) o;
      return row == other.row && col == other.col;
    }

    @Override
    public int hashCode() {
      return row + 31 * col;
    }

    @Override
    public void write(DataOutput out) throws IOException {
      out.writeInt(row);
      out.writeInt(col);
      out.writeDouble(val);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
      row = in.readInt();
      col = in.readInt();
      val = in.readDouble();
    }

    @Override
    public String toString() {
      return "(" + row + ',' + col + "):" + val;
    }
  }
}
