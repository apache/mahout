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
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorIterable;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.decomposer.SolverTest;
import org.apache.mahout.math.function.Functions;
import org.junit.Test;

import com.google.common.base.Function;
import com.google.common.collect.Iterators;
import com.google.common.collect.Maps;

public final class TestDistributedRowMatrix extends MahoutTestCase {
  public static final String TEST_PROPERTY_KEY = "test.property.key";
  public static final String TEST_PROPERTY_VALUE = "test.property.value";

  private static void assertEquals(VectorIterable m, VectorIterable mtt, double errorTolerance) {
    Iterator<MatrixSlice> mIt = m.iterateAll();
    Iterator<MatrixSlice> mttIt = mtt.iterateAll();
    Map<Integer, Vector> mMap = Maps.newHashMap();
    Map<Integer, Vector> mttMap = Maps.newHashMap();
    while (mIt.hasNext() && mttIt.hasNext()) {
      MatrixSlice ms = mIt.next();
      mMap.put(ms.index(), ms.vector());
      MatrixSlice mtts = mttIt.next();
      mttMap.put(mtts.index(), mtts.vector());
    }
    for (Map.Entry<Integer, Vector> entry : mMap.entrySet()) {
      Integer key = entry.getKey();
      Vector value = entry.getValue();
      if (value == null || mttMap.get(key) == null) {
        assertTrue(value == null || value.norm(2) == 0);
        assertTrue(mttMap.get(key) == null || mttMap.get(key).norm(2) == 0);
      } else {
        assertTrue(
            value.getDistanceSquared(mttMap.get(key)) < errorTolerance);
      }
    }
  }

  @Test
  public void testTranspose() throws Exception {
    DistributedRowMatrix m = randomDistributedMatrix(10, 9, 5, 4, 1.0, false);
    m.setConf(getConfiguration());
    DistributedRowMatrix mt = m.transpose();
    mt.setConf(getConfiguration());

    Path tmpPath = getTestTempDirPath();
    m.setOutputTempPathString(tmpPath.toString());
    Path tmpOutPath = new Path(tmpPath, "/tmpOutTranspose");
    mt.setOutputTempPathString(tmpOutPath.toString());
    HadoopUtil.delete(getConfiguration(), tmpOutPath);
    DistributedRowMatrix mtt = mt.transpose();
    assertEquals(m, mtt, EPSILON);
  }

  @Test
  public void testMatrixColumnMeansJob() throws Exception {
    Matrix m =
        SolverTest.randomSequentialAccessSparseMatrix(100, 90, 50, 20, 1.0);
    DistributedRowMatrix dm =
        randomDistributedMatrix(100, 90, 50, 20, 1.0, false);
    dm.setConf(getConfiguration());

    Vector expected = new DenseVector(50);
    for (int i = 0; i < m.numRows(); i++) {
      expected.assign(m.viewRow(i), Functions.PLUS);
    }
    expected.assign(Functions.DIV, m.numRows());
    Vector actual = dm.columnMeans("DenseVector");
    assertEquals(0.0, expected.getDistanceSquared(actual), EPSILON);
  }

  @Test
  public void testNullMatrixColumnMeansJob() throws Exception {
    Matrix m =
        SolverTest.randomSequentialAccessSparseMatrix(100, 90, 0, 0, 1.0);
    DistributedRowMatrix dm =
        randomDistributedMatrix(100, 90, 0, 0, 1.0, false);
    dm.setConf(getConfiguration());

    Vector expected = new DenseVector(0);
    for (int i = 0; i < m.numRows(); i++) {
      expected.assign(m.viewRow(i), Functions.PLUS);
    }
    expected.assign(Functions.DIV, m.numRows());
    Vector actual = dm.columnMeans();
    assertEquals(0.0, expected.getDistanceSquared(actual), EPSILON);
  }

  @Test
  public void testMatrixTimesVector() throws Exception {
    Vector v = new RandomAccessSparseVector(50);
    v.assign(1.0);
    Matrix m = SolverTest.randomSequentialAccessSparseMatrix(100, 90, 50, 20, 1.0);
    DistributedRowMatrix dm = randomDistributedMatrix(100, 90, 50, 20, 1.0, false);
    dm.setConf(getConfiguration());

    Vector expected = m.times(v);
    Vector actual = dm.times(v);
    assertEquals(0.0, expected.getDistanceSquared(actual), EPSILON);
  }

  @Test
  public void testMatrixTimesSquaredVector() throws Exception {
    Vector v = new RandomAccessSparseVector(50);
    v.assign(1.0);
    Matrix m = SolverTest.randomSequentialAccessSparseMatrix(100, 90, 50, 20, 1.0);
    DistributedRowMatrix dm = randomDistributedMatrix(100, 90, 50, 20, 1.0, false);
    dm.setConf(getConfiguration());

    Vector expected = m.timesSquared(v);
    Vector actual = dm.timesSquared(v);
    assertEquals(0.0, expected.getDistanceSquared(actual), 1.0e-9);
  }

  @Test
  public void testMatrixTimesMatrix() throws Exception {
    Matrix inputA = SolverTest.randomSequentialAccessSparseMatrix(20, 19, 15, 5, 10.0);
    Matrix inputB = SolverTest.randomSequentialAccessSparseMatrix(20, 13, 25, 10, 5.0);
    Matrix expected = inputA.transpose().times(inputB);

    DistributedRowMatrix distA = randomDistributedMatrix(20, 19, 15, 5, 10.0, false, "distA");
    distA.setConf(getConfiguration());
    DistributedRowMatrix distB = randomDistributedMatrix(20, 13, 25, 10, 5.0, false, "distB");
    distB.setConf(getConfiguration());
    DistributedRowMatrix product = distA.times(distB);

    assertEquals(expected, product, EPSILON);
  }

  @Test
  public void testMatrixMultiplactionJobConfBuilder() throws Exception {
    Configuration initialConf = createInitialConf();

    Path baseTmpDirPath = getTestTempDirPath("testpaths");
    Path aPath = new Path(baseTmpDirPath, "a");
    Path bPath = new Path(baseTmpDirPath, "b");
    Path outPath = new Path(baseTmpDirPath, "out");

    Configuration mmJobConf = MatrixMultiplicationJob.createMatrixMultiplyJobConf(aPath, bPath, outPath, 10);
    Configuration mmCustomJobConf = MatrixMultiplicationJob.createMatrixMultiplyJobConf(initialConf,
                                                                                        aPath,
                                                                                        bPath,
                                                                                        outPath,
                                                                                        10);

    assertNull(mmJobConf.get(TEST_PROPERTY_KEY));
    assertEquals(TEST_PROPERTY_VALUE, mmCustomJobConf.get(TEST_PROPERTY_KEY));
  }

  @Test
  public void testTransposeJobConfBuilder() throws Exception {
    Configuration initialConf = createInitialConf();

    Path baseTmpDirPath = getTestTempDirPath("testpaths");
    Path inputPath = new Path(baseTmpDirPath, "input");
    Path outputPath = new Path(baseTmpDirPath, "output");

    Configuration transposeJobConf = TransposeJob.buildTransposeJobConf(inputPath, outputPath, 10);
    Configuration transposeCustomJobConf = TransposeJob.buildTransposeJobConf(initialConf, inputPath, outputPath, 10);

    assertNull(transposeJobConf.get(TEST_PROPERTY_KEY));
    assertEquals(TEST_PROPERTY_VALUE, transposeCustomJobConf.get(TEST_PROPERTY_KEY));
  }

  @Test public void testTimesSquaredJobConfBuilders() throws Exception {
    Configuration initialConf = createInitialConf();

    Path baseTmpDirPath = getTestTempDirPath("testpaths");
    Path inputPath = new Path(baseTmpDirPath, "input");
    Path outputPath = new Path(baseTmpDirPath, "output");

    Vector v = new RandomAccessSparseVector(50);
    v.assign(1.0);

    Configuration timesSquaredJobConf1 = TimesSquaredJob.createTimesSquaredJobConf(v, inputPath, outputPath);
    Configuration customTimesSquaredJobConf1 = TimesSquaredJob.createTimesSquaredJobConf(initialConf, v, inputPath, outputPath);

    assertNull(timesSquaredJobConf1.get(TEST_PROPERTY_KEY));
    assertEquals(TEST_PROPERTY_VALUE, customTimesSquaredJobConf1.get(TEST_PROPERTY_KEY));

    Configuration timesJobConf = TimesSquaredJob.createTimesJobConf(v, 50, inputPath, outputPath);
    Configuration customTimesJobConf = TimesSquaredJob.createTimesJobConf(initialConf, v, 50, inputPath, outputPath);

    assertNull(timesJobConf.get(TEST_PROPERTY_KEY));
    assertEquals(TEST_PROPERTY_VALUE, customTimesJobConf.get(TEST_PROPERTY_KEY));

    Configuration timesSquaredJobConf2 = TimesSquaredJob.createTimesSquaredJobConf(v,
                                                                                   inputPath,
                                                                                   outputPath,
                                                                                   TimesSquaredJob.TimesSquaredMapper.class,
                                                                                   TimesSquaredJob.VectorSummingReducer.class);
    Configuration customTimesSquaredJobConf2 = TimesSquaredJob.createTimesSquaredJobConf(initialConf,
                                                                                         v,
                                                                                         inputPath,
                                                                                         outputPath,
                                                                                         TimesSquaredJob.TimesSquaredMapper.class,
                                                                                         TimesSquaredJob.VectorSummingReducer.class);

    assertNull(timesSquaredJobConf2.get(TEST_PROPERTY_KEY));
    assertEquals(TEST_PROPERTY_VALUE, customTimesSquaredJobConf2.get(TEST_PROPERTY_KEY));

    Configuration timesSquaredJobConf3 = TimesSquaredJob.createTimesSquaredJobConf(v,
                                                                                   50,
                                                                                   inputPath,
                                                                                   outputPath,
                                                                                   TimesSquaredJob.TimesSquaredMapper.class,
                                                                                   TimesSquaredJob.VectorSummingReducer.class);
    Configuration customTimesSquaredJobConf3 = TimesSquaredJob.createTimesSquaredJobConf(initialConf,
                                                                                         v,
                                                                                         50,
                                                                                         inputPath,
                                                                                         outputPath,
                                                                                         TimesSquaredJob.TimesSquaredMapper.class,
                                                                                         TimesSquaredJob.VectorSummingReducer.class);

    assertNull(timesSquaredJobConf3.get(TEST_PROPERTY_KEY));
    assertEquals(TEST_PROPERTY_VALUE, customTimesSquaredJobConf3.get(TEST_PROPERTY_KEY));
  }

  @Test
  public void testTimesVectorTempDirDeletion() throws Exception {
    Configuration conf = getConfiguration();
    Vector v = new RandomAccessSparseVector(50);
    v.assign(1.0);
    DistributedRowMatrix dm = randomDistributedMatrix(100, 90, 50, 20, 1.0, false);
    dm.setConf(conf);

    Path outputPath = dm.getOutputTempPath();
    FileSystem fs = outputPath.getFileSystem(conf);

    deleteContentsOfPath(conf, outputPath);

    assertEquals(0, HadoopUtil.listStatus(fs, outputPath).length);

    Vector result1 = dm.times(v);

    assertEquals(0, HadoopUtil.listStatus(fs, outputPath).length);

    deleteContentsOfPath(conf, outputPath);
    assertEquals(0, HadoopUtil.listStatus(fs, outputPath).length);

    conf.setBoolean(DistributedRowMatrix.KEEP_TEMP_FILES, true);
    dm.setConf(conf);

    Vector result2 = dm.times(v);

    FileStatus[] outputStatuses = fs.listStatus(outputPath);
    assertEquals(1, outputStatuses.length);
    Path outputTempPath = outputStatuses[0].getPath();
    Path inputVectorPath = new Path(outputTempPath, TimesSquaredJob.INPUT_VECTOR);
    Path outputVectorPath = new Path(outputTempPath, TimesSquaredJob.OUTPUT_VECTOR_FILENAME);
    assertEquals(1, fs.listStatus(inputVectorPath, PathFilters.logsCRCFilter()).length);
    assertEquals(1, fs.listStatus(outputVectorPath, PathFilters.logsCRCFilter()).length);

    assertEquals(0.0, result1.getDistanceSquared(result2), EPSILON);
  }

  @Test
  public void testTimesSquaredVectorTempDirDeletion() throws Exception {
    Configuration conf = getConfiguration();
    Vector v = new RandomAccessSparseVector(50);
    v.assign(1.0);
    DistributedRowMatrix dm = randomDistributedMatrix(100, 90, 50, 20, 1.0, false);
    dm.setConf(getConfiguration());

    Path outputPath = dm.getOutputTempPath();
    FileSystem fs = outputPath.getFileSystem(conf);

    deleteContentsOfPath(conf, outputPath);

    assertEquals(0, HadoopUtil.listStatus(fs, outputPath).length);

    Vector result1 = dm.timesSquared(v);

    assertEquals(0, HadoopUtil.listStatus(fs, outputPath).length);

    deleteContentsOfPath(conf, outputPath);
    assertEquals(0, HadoopUtil.listStatus(fs, outputPath).length);

    conf.setBoolean(DistributedRowMatrix.KEEP_TEMP_FILES, true);
    dm.setConf(conf);

    Vector result2 = dm.timesSquared(v);

    FileStatus[] outputStatuses = fs.listStatus(outputPath);
    assertEquals(1, outputStatuses.length);
    Path outputTempPath = outputStatuses[0].getPath();
    Path inputVectorPath = new Path(outputTempPath, TimesSquaredJob.INPUT_VECTOR);
    Path outputVectorPath = new Path(outputTempPath, TimesSquaredJob.OUTPUT_VECTOR_FILENAME);
    assertEquals(1, fs.listStatus(inputVectorPath, PathFilters.logsCRCFilter()).length);
    assertEquals(1, fs.listStatus(outputVectorPath, PathFilters.logsCRCFilter()).length);

    assertEquals(0.0, result1.getDistanceSquared(result2), EPSILON);
  }

  public Configuration createInitialConf() throws IOException {
    Configuration initialConf = getConfiguration();
    initialConf.set(TEST_PROPERTY_KEY, TEST_PROPERTY_VALUE);
    return initialConf;
  }

  private static void deleteContentsOfPath(Configuration conf, Path path) throws Exception {
    FileSystem fs = path.getFileSystem(conf);

    FileStatus[] statuses = HadoopUtil.listStatus(fs, path);
    for (FileStatus status : statuses) {
      fs.delete(status.getPath(), true);
    }
  }

  public DistributedRowMatrix randomDistributedMatrix(int numRows,
                                                      int nonNullRows,
                                                      int numCols,
                                                      int entriesPerRow,
                                                      double entryMean,
                                                      boolean isSymmetric) throws IOException {
    return randomDistributedMatrix(numRows, nonNullRows, numCols, entriesPerRow, entryMean, isSymmetric, "testdata");
  }

  public DistributedRowMatrix randomDenseHierarchicalDistributedMatrix(int numRows,
                                                                       int numCols,
                                                                       boolean isSymmetric,
                                                                       String baseTmpDirSuffix)
    throws IOException {
    Path baseTmpDirPath = getTestTempDirPath(baseTmpDirSuffix);
    Matrix c = SolverTest.randomHierarchicalMatrix(numRows, numCols, isSymmetric);
    return saveToFs(c, baseTmpDirPath);
  }

  public DistributedRowMatrix randomDistributedMatrix(int numRows,
                                                      int nonNullRows,
                                                      int numCols,
                                                      int entriesPerRow,
                                                      double entryMean,
                                                      boolean isSymmetric,
                                                      String baseTmpDirSuffix) throws IOException {
    Path baseTmpDirPath = getTestTempDirPath(baseTmpDirSuffix);
    Matrix c = SolverTest.randomSequentialAccessSparseMatrix(numRows, nonNullRows, numCols, entriesPerRow, entryMean);
    if (isSymmetric) {
      c = c.times(c.transpose());
    }
    return saveToFs(c, baseTmpDirPath);
  }

  private DistributedRowMatrix saveToFs(final Matrix m, Path baseTmpDirPath) throws IOException {
    Configuration conf = getConfiguration();
    FileSystem fs = FileSystem.get(baseTmpDirPath.toUri(), conf);

    ClusteringTestUtils.writePointsToFile(new Iterable<VectorWritable>() {
      @Override
      public Iterator<VectorWritable> iterator() {
        return Iterators.transform(m.iterator(), new Function<MatrixSlice,VectorWritable>() {
          @Override
          public VectorWritable apply(MatrixSlice input) {
            return new VectorWritable(input.vector());
          }
        });
      }
    }, true, new Path(baseTmpDirPath, "distMatrix/part-00000"), fs, conf);

    DistributedRowMatrix distMatrix = new DistributedRowMatrix(new Path(baseTmpDirPath, "distMatrix"),
                                                               new Path(baseTmpDirPath, "tmpOut"),
                                                               m.numRows(),
                                                               m.numCols());
    distMatrix.setConf(new Configuration(conf));

    return distMatrix;
  }
}
