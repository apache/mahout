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

package org.apache.mahout.cf.taste.hadoop.als;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.hadoop.MathHelper;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Iterator;

public class ParallelALSFactorizationJobTest extends TasteTestCase {

  private static final Logger log = LoggerFactory.getLogger(ParallelALSFactorizationJobTest.class);

  private File inputFile;
  private File outputDir;
  private File tmpDir;
  private Configuration conf;

  @Before
  @Override
  public void setUp() throws Exception {
    super.setUp();
    inputFile = getTestTempFile("prefs.txt");
    outputDir = getTestTempDir("output");
    outputDir.delete();
    tmpDir = getTestTempDir("tmp");

    conf = new Configuration();
  }

  /**
   * small integration test that runs the full job
   *
   * <pre>
   *
   *  user-item-matrix
   *
   *          burger  hotdog  berries  icecream
   *  dog       5       5        2        -
   *  rabbit    2       -        3        5
   *  cow       -       5        -        3
   *  donkey    3       -        -        5
   *
   * </pre>
   */
  @Test
  public void completeJobToyExample() throws Exception {

    Double na = Double.NaN;
    Matrix preferences = new SparseRowMatrix(4, 4, new Vector[] {
        new DenseVector(new double[] { 5.0, 5.0, 2.0, na }),
        new DenseVector(new double[] { 2.0, na,  3.0, 5.0 }),
        new DenseVector(new double[] { na,  5.0, na,  3.0 }),
        new DenseVector(new double[] { 3.0, na,  na,  5.0 }) });

    writeLines(inputFile, preferencesAsText(preferences));

    ParallelALSFactorizationJob alsFactorization = new ParallelALSFactorizationJob();
    alsFactorization.setConf(conf);

    int numFeatures = 3;
    int numIterations = 5;
    double lambda = 0.065;

    alsFactorization.run(new String[] { "--input", inputFile.getAbsolutePath(), "--output", outputDir.getAbsolutePath(),
        "--tempDir", tmpDir.getAbsolutePath(), "--lambda", String.valueOf(lambda),
        "--numFeatures", String.valueOf(numFeatures), "--numIterations", String.valueOf(numIterations) });

    Matrix u = MathHelper.readMatrix(conf, new Path(outputDir.getAbsolutePath(), "U/part-m-00000"),
        preferences.numRows(), numFeatures);
    Matrix m = MathHelper.readMatrix(conf, new Path(outputDir.getAbsolutePath(), "M/part-m-00000"),
        preferences.numCols(), numFeatures);

    StringBuilder info = new StringBuilder();
    info.append("\nA - users x items\n\n");
    info.append(MathHelper.nice(preferences));
    info.append("\nU - users x features\n\n");
    info.append(MathHelper.nice(u));
    info.append("\nM - items x features\n\n");
    info.append(MathHelper.nice(m));
    Matrix Ak = u.times(m.transpose());
    info.append("\nAk - users x items\n\n");
    info.append(MathHelper.nice(Ak));
    info.append('\n');

    log.info(info.toString());

    RunningAverage avg = new FullRunningAverage();
    Iterator<MatrixSlice> sliceIterator = preferences.iterateAll();
    while (sliceIterator.hasNext()) {
      MatrixSlice slice = sliceIterator.next();
      Iterator<Vector.Element> elementIterator = slice.vector().iterateNonZero();
      while (elementIterator.hasNext()) {
        Vector.Element e = elementIterator.next();
        if (!Double.isNaN(e.get())) {
          double pref = e.get();
          double estimate = u.viewRow(slice.index()).dot(m.viewRow(e.index()));
          double err = pref - estimate;
          avg.addDatum(err * err);
          log.info("Comparing preference of user [{}] towards item [{}], was [{}] estimate is [{}]",
                   new Object[] { slice.index(), e.index(), pref, estimate });
        }
      }
    }
    double rmse = Math.sqrt(avg.getAverage());
    log.info("RMSE: {}", rmse);

    assertTrue(rmse < 0.2);
  }

  @Test
  public void completeJobImplicitToyExample() throws Exception {

    Matrix observations = new SparseRowMatrix(4, 4, new Vector[] {
        new DenseVector(new double[] { 5.0, 5.0, 2.0, 0 }),
        new DenseVector(new double[] { 2.0, 0,   3.0, 5.0 }),
        new DenseVector(new double[] { 0,   5.0, 0,   3.0 }),
        new DenseVector(new double[] { 3.0, 0,   0,   5.0 }) });

    Matrix preferences = new SparseRowMatrix(4, 4, new Vector[] {
        new DenseVector(new double[] { 1.0, 1.0, 1.0, 0 }),
        new DenseVector(new double[] { 1.0, 0,   1.0, 1.0 }),
        new DenseVector(new double[] { 0,   1.0, 0,   1.0 }),
        new DenseVector(new double[] { 1.0, 0,   0,   1.0 }) });

    writeLines(inputFile, preferencesAsText(observations));

    ParallelALSFactorizationJob alsFactorization = new ParallelALSFactorizationJob();
    alsFactorization.setConf(conf);

    int numFeatures = 3;
    int numIterations = 5;
    double lambda = 0.065;
    double alpha = 20;

    alsFactorization.run(new String[] { "--input", inputFile.getAbsolutePath(), "--output", outputDir.getAbsolutePath(),
        "--tempDir", tmpDir.getAbsolutePath(), "--lambda", String.valueOf(lambda),
        "--implicitFeedback", String.valueOf(true), "--alpha", String.valueOf(alpha),
        "--numFeatures", String.valueOf(numFeatures), "--numIterations", String.valueOf(numIterations) });

    Matrix u = MathHelper.readMatrix(conf, new Path(outputDir.getAbsolutePath(), "U/part-m-00000"),
        observations.numRows(), numFeatures);
    Matrix m = MathHelper.readMatrix(conf, new Path(outputDir.getAbsolutePath(), "M/part-m-00000"),
        observations.numCols(), numFeatures);

    StringBuilder info = new StringBuilder();
    info.append("\nObservations - users x items\n");
    info.append(MathHelper.nice(observations));
    info.append("\nA - users x items\n\n");
    info.append(MathHelper.nice(preferences));
    info.append("\nU - users x features\n\n");
    info.append(MathHelper.nice(u));
    info.append("\nM - items x features\n\n");
    info.append(MathHelper.nice(m));
    Matrix Ak = u.times(m.transpose());
    info.append("\nAk - users x items\n\n");
    info.append(MathHelper.nice(Ak));
    info.append('\n');

    log.info(info.toString());

    RunningAverage avg = new FullRunningAverage();
    Iterator<MatrixSlice> sliceIterator = preferences.iterateAll();
    while (sliceIterator.hasNext()) {
      MatrixSlice slice = sliceIterator.next();
      for (Vector.Element e : slice.vector()) {
        if (!Double.isNaN(e.get())) {
          double pref = e.get();
          double estimate = u.viewRow(slice.index()).dot(m.viewRow(e.index()));
          double confidence = 1 + alpha * observations.getQuick(slice.index(), e.index());
          double err = confidence * (pref - estimate) * (pref - estimate);
          avg.addDatum(err);
          log.info("Comparing preference of user [{}] towards item [{}], was [{}] with confidence [{}] " 
                       + "estimate is [{}]", new Object[]{slice.index(), e.index(), pref, confidence, estimate});
        }
      }
    }
    double rmse = Math.sqrt(avg.getAverage());
    log.info("RMSE: {}", rmse);

    assertTrue(rmse < 0.4);
  }

  protected static String preferencesAsText(Matrix preferences) {
    StringBuilder prefsAsText = new StringBuilder();
    String separator = "";
    Iterator<MatrixSlice> sliceIterator = preferences.iterateAll();
    while (sliceIterator.hasNext()) {
      MatrixSlice slice = sliceIterator.next();
      Iterator<Vector.Element> elementIterator = slice.vector().iterateNonZero();
      while (elementIterator.hasNext()) {
        Vector.Element e = elementIterator.next();
        if (!Double.isNaN(e.get())) {
          prefsAsText.append(separator).append(slice.index()).append(',').append(e.index()).append(',').append(e.get());
          separator = "\n";
        }
      }
    }
    return prefsAsText.toString();
  }


}
