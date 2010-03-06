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

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.JobConfigurable;
import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorIterable;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Iterator;
import java.util.NoSuchElementException;


/**
 * DistributedRowMatrix is a FileSystem-backed VectorIterable in which the vectors live in a
 * SequenceFile<WritableComparable,VectorWritable>, and distributed operations are executed as M/R passes on
 * Hadoop.  The usage is as follows: <p>
 * <p>
 * <pre>
 *   // the path must already contain an already created SequenceFile!
 *   DistributedRowMatrix m = new DistributedRowMatrix("path/to/vector/sequenceFile", "tmp/path", 10000000, 250000);
 *   m.configure(new JobConf());
 *   // now if we want to multiply a vector by this matrix, it's dimension must equal the row dimension of this
 *   // matrix.  If we want to timesSquared() a vector by this matrix, its dimension must equal the column dimension
 *   // of the matrix.
 *   Vector v = new DenseVector(250000);
 *   // now the following operation will be done via a M/R pass via Hadoop.
 *   Vector w = m.timesSquared(v);
 * </pre>
 *
 */
public class DistributedRowMatrix implements VectorIterable, JobConfigurable {

  private static final Logger log = LoggerFactory.getLogger(DistributedRowMatrix.class);

  private final String inputPathString;
  private String outputTmpPathString;
  private JobConf conf;
  private Path rowPath;
  private Path outputTmpBasePath;
  private final int numRows;
  private final int numCols;

  @Override
  public void configure(JobConf conf) {
    this.conf = conf;
    try {
      rowPath= FileSystem.get(conf).makeQualified(new Path(inputPathString));
      outputTmpBasePath = FileSystem.get(conf).makeQualified(new Path(outputTmpPathString));
    } catch(IOException ioe) {
      throw new IllegalStateException(ioe);
    }
  }

  public DistributedRowMatrix(String inputPathString,
                              String outputTmpPathString,
                              int numRows,
                              int numCols) {
    this.inputPathString = inputPathString;
    this.outputTmpPathString = outputTmpPathString;
    this.numRows = numRows;
    this.numCols = numCols;
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
          outPathString, outputTmpBasePath.toString());
    }
  }

  @Override
  public Iterator<MatrixSlice> iterateAll() {
    try {
      FileSystem fs = FileSystem.get(conf);
      return new DistributedMatrixIterator(fs, rowPath, conf);
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

  public DistributedRowMatrix times(DistributedRowMatrix other) {
    if(numRows != other.numRows()) {
      throw new CardinalityException(numRows, other.numRows());
    }
    Path outPath = new Path(outputTmpBasePath.getParent(), "productWith");
    JobConf conf = MatrixMultiplicationJob.createMatrixMultiplyJobConf(rowPath, other.rowPath, outPath, other.numCols);
    try {
      JobClient.runJob(conf);
      DistributedRowMatrix out = new DistributedRowMatrix(outPath.toString(),
          outputTmpPathString, numRows, other.numCols());
      out.configure(conf);
      return out;
    } catch (IOException ioe) {
      throw new RuntimeException(ioe);
    }
  }

  public DistributedRowMatrix transpose() {
    Path outputPath = new Path(rowPath.getParent(), "transpose-" + (byte)System.nanoTime());
    try {
      JobConf conf = TransposeJob.buildTransposeJobConf(rowPath, outputPath, numRows);
      JobClient.runJob(conf);
      DistributedRowMatrix m = new DistributedRowMatrix(outputPath.toString(), outputTmpPathString, numCols, numRows);
      m.configure(this.conf);
      return m;
    } catch (IOException ioe) {
      throw new RuntimeException(ioe);
    }
  }

  @Override
  public Vector times(Vector v) {
    try {
      JobConf conf = TimesSquaredJob.createTimesJobConf(v,
                                                        numRows,
                                                        rowPath,
                                                        new Path(outputTmpPathString,
                                                                 new Path(Long.toString(System.nanoTime()))));
      JobClient.runJob(conf);
      return TimesSquaredJob.retrieveTimesSquaredOutputVector(conf);
    } catch(IOException ioe) {
      throw new RuntimeException(ioe);
    }
  }

  @Override
  public Vector timesSquared(Vector v) {
    try {
      JobConf conf = TimesSquaredJob.createTimesSquaredJobConf(v,
                                                               rowPath,
                                                               new Path(outputTmpBasePath,
                                                                        new Path(Long.toString(System.nanoTime()))));
      JobClient.runJob(conf);
      return TimesSquaredJob.retrieveTimesSquaredOutputVector(conf);
    } catch(IOException ioe) {
      throw new IllegalStateException(ioe);
    }
  }
  
  @Override
  public Iterator<MatrixSlice> iterator() {
    return iterateAll();
  }

  public static class DistributedMatrixIterator implements Iterator<MatrixSlice> {
    private SequenceFile.Reader reader;
    private FileStatus[] statuses;
    private boolean hasBuffered = false;
    private boolean hasNext = false;
    private int statusIndex = 0;
    private final FileSystem fs;
    private final JobConf conf;
    private final IntWritable i = new IntWritable();
    private final VectorWritable v = new VectorWritable();

    public DistributedMatrixIterator(FileSystem fs, Path rowPath, JobConf conf) throws IOException {
      this.fs = fs;
      this.conf = conf;
      statuses = fs.globStatus(new Path(rowPath, "*"));
      reader = new SequenceFile.Reader(fs, statuses[statusIndex].getPath(), conf);
    }

    @Override
    public boolean hasNext() {
      try {
        if(!hasBuffered) {
          hasNext = reader.next(i, v);
          if(!hasNext && statusIndex < statuses.length - 1) {
            statusIndex++;
            reader = new SequenceFile.Reader(fs, statuses[statusIndex].getPath(), conf);
            hasNext = reader.next(i, v);
          }
          hasBuffered = true;
        }
      } catch (IOException ioe) {
        throw new IllegalStateException(ioe);
      } finally {
        if(!hasNext) {
          try { reader.close(); } catch (IOException ioe) {}
        }
      }
      return hasNext;

    }

    @Override
    public MatrixSlice next() {
      if(!hasBuffered && !hasNext()) {
        throw new NoSuchElementException();
      }
      hasBuffered = false;
      return new MatrixSlice(v.get(), i.get());
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException("Cannot remove from DistributedMatrixIterator");
    }
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
      if(row > o.row) {
        return 1;
      } else if(row < o.row) {
        return -1;
      } else {
        if(col > o.col) {
          return 1;
        } else if(col < o.col) {
          return -1;
        } else {
          return 0;
        }
      }
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
  }

}
