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
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Iterator;

public class ParallelALSFactorizationJobTest extends TasteTestCase {

  private static final Logger log = LoggerFactory.getLogger(ParallelALSFactorizationJobTest.class);


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

    File inputFile = getTestTempFile("prefs.txt");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File tmpDir = getTestTempDir("tmp");

    Double na = Double.NaN;
    Matrix preferences = new SparseRowMatrix(4, 4, new Vector[] {
        new DenseVector(new double[] { 5.0, 5.0, 2.0, na }),
        new DenseVector(new double[] { 2.0, na,  3.0, 5.0 }),
        new DenseVector(new double[] { na,  5.0, na,  3.0 }),
        new DenseVector(new double[] { 3.0, na,  na,  5.0 }) });

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
    log.info("Input matrix:\n{}", prefsAsText);
    writeLines(inputFile, prefsAsText.toString());

    ParallelALSFactorizationJob alsFactorization = new ParallelALSFactorizationJob();

    Configuration conf = new Configuration();
    conf.set("mapred.input.dir", inputFile.getAbsolutePath());
    conf.set("mapred.output.dir", outputDir.getAbsolutePath());
    conf.setBoolean("mapred.output.compress", false);

    alsFactorization.setConf(conf);
    int numFeatures = 3;
    int numIterations = 5;
    double lambda = 0.065;
    alsFactorization.run(new String[] { "--tempDir", tmpDir.getAbsolutePath(), "--lambda", String.valueOf(lambda),
        "--numFeatures", String.valueOf(numFeatures), "--numIterations", String.valueOf(numIterations) });

    Matrix u = MathHelper.readMatrix(conf, new Path(outputDir.getAbsolutePath(), "U/part-m-00000"),
        preferences.numRows(), numFeatures);
    Matrix m = MathHelper.readMatrix(conf, new Path(outputDir.getAbsolutePath(), "M/part-m-00000"),
        preferences.numCols(), numFeatures);

    System.out.println("A - users x items\n");
    for (int n = 0; n < preferences.numRows(); n++) {
      System.out.println(ALSUtils.nice(preferences.viewRow(n)));
    }
    System.out.println("\nU - users x features\n");
    for (int n = 0; n < u.numRows(); n++) {
      System.out.println(ALSUtils.nice(u.viewRow(n)));
    }
    System.out.println("\nM - items x features\n");
    for (int n = 0; n < m.numRows(); n++) {
      System.out.println(ALSUtils.nice(m.viewRow(n)));
    }
    Matrix Ak = u.times(m.transpose());
    System.out.println("\nAk - users x items\n");
    for (int n = 0; n < Ak.numRows(); n++) {
      System.out.println(ALSUtils.nice(Ak.viewRow(n)));
    }

    System.out.println();


    RunningAverage avg = new FullRunningAverage();
    sliceIterator = preferences.iterateAll();
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

}
