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
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.als.AlternateLeastSquaresSolver;
import org.apache.mahout.math.hadoop.MathHelper;
import org.easymock.IArgumentMatcher;
import org.easymock.EasyMock;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.Iterator;

public class ParallelALSFactorizationJobTest extends TasteTestCase {

  private static final Logger log = LoggerFactory.getLogger(ParallelALSFactorizationJobTest.class);

  @Test
  public void prefsToRatingsMapper() throws Exception {
    Mapper<LongWritable,Text,VarIntWritable,FeatureVectorWithRatingWritable>.Context ctx =
      EasyMock.createMock(Mapper.Context.class);
    ctx.write(new VarIntWritable(TasteHadoopUtils.idToIndex(456L)),
        new FeatureVectorWithRatingWritable(TasteHadoopUtils.idToIndex(123L), 2.35f));
    EasyMock.replay(ctx);

    new ParallelALSFactorizationJob.PrefsToRatingsMapper().map(null, new Text("123,456,2.35"), ctx);
    EasyMock.verify(ctx);
  }

  @Test
  public void prefsToRatingsMapperTranspose() throws Exception {
    Mapper<LongWritable,Text,VarIntWritable,FeatureVectorWithRatingWritable>.Context ctx =
      EasyMock.createMock(Mapper.Context.class);
    ctx.write(new VarIntWritable(TasteHadoopUtils.idToIndex(123L)),
        new FeatureVectorWithRatingWritable(TasteHadoopUtils.idToIndex(456L), 2.35f));
    EasyMock.replay(ctx);

    ParallelALSFactorizationJob.PrefsToRatingsMapper mapper = new ParallelALSFactorizationJob.PrefsToRatingsMapper();
    setField(mapper, "transpose", true);
    mapper.map(null, new Text("123,456,2.35"), ctx);
    EasyMock.verify(ctx);
  }

  @Test
  public void initializeMReducer() throws Exception {
    Reducer<VarLongWritable,FloatWritable,VarIntWritable,FeatureVectorWithRatingWritable>.Context ctx =
        EasyMock.createMock(Reducer.Context.class);
    ctx.write(EasyMock.eq(new VarIntWritable(TasteHadoopUtils.idToIndex(123L))), matchInitializedFeatureVector(3.0, 3));
    EasyMock.replay(ctx);

    ParallelALSFactorizationJob.InitializeMReducer reducer = new ParallelALSFactorizationJob.InitializeMReducer();
    setField(reducer, "numFeatures", 3);
    reducer.reduce(new VarLongWritable(123L), Arrays.asList(new FloatWritable(4.0f), new FloatWritable(2.0f)), ctx);
    EasyMock.verify(ctx);
  }

  static FeatureVectorWithRatingWritable matchInitializedFeatureVector(final double average, final int numFeatures) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof FeatureVectorWithRatingWritable) {
          Vector v = ((FeatureVectorWithRatingWritable) argument).getFeatureVector();
          if (v.get(0) != average) {
            return false;
          }
          for (int n = 1; n < numFeatures; n++) {
            if (v.get(n) < 0 || v.get(n) > 1) {
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

  @Test
  public void itemIDRatingMapper() throws Exception {
    Mapper<LongWritable,Text,VarLongWritable,FloatWritable>.Context ctx = EasyMock.createMock(Mapper.Context.class);
    ctx.write(new VarLongWritable(456L), new FloatWritable(2.35f));
    EasyMock.replay(ctx);
    new ParallelALSFactorizationJob.ItemIDRatingMapper().map(null, new Text("123,456,2.35"), ctx);
    EasyMock.verify(ctx);
  }

  @Test
  public void joinFeatureVectorAndRatingsReducer() throws Exception {
    Vector vector = new DenseVector(new double[] { 4.5, 1.2 });
    Reducer<VarIntWritable,FeatureVectorWithRatingWritable,IndexedVarIntWritable,FeatureVectorWithRatingWritable>.Context ctx =
        EasyMock.createMock(Reducer.Context.class);
    ctx.write(new IndexedVarIntWritable(456, 123), new FeatureVectorWithRatingWritable(123, 2.35f, vector));
    EasyMock.replay(ctx);
    new ParallelALSFactorizationJob.JoinFeatureVectorAndRatingsReducer().reduce(new VarIntWritable(123),
        Arrays.asList(new FeatureVectorWithRatingWritable(456, vector),
        new FeatureVectorWithRatingWritable(456, 2.35f)), ctx);
    EasyMock.verify(ctx);
  }


  @Test
  public void solvingReducer() throws Exception {

    AlternateLeastSquaresSolver solver = new AlternateLeastSquaresSolver();

    int numFeatures = 2;
    double lambda = 0.01;
    Vector ratings = new DenseVector(new double[] { 2, 1 });
    Vector col1 = new DenseVector(new double[] { 1, 2 });
    Vector col2 = new DenseVector(new double[] { 3, 4 });

    Vector result = solver.solve(Arrays.asList(col1, col2), ratings, lambda, numFeatures);
    Vector.Element[] elems = new Vector.Element[result.size()];
    for (int n = 0; n < result.size(); n++) {
      elems[n] = result.getElement(n);
    }

    Reducer<IndexedVarIntWritable,FeatureVectorWithRatingWritable,VarIntWritable,FeatureVectorWithRatingWritable>.Context ctx =
        EasyMock.createMock(Reducer.Context.class);
    ctx.write(EasyMock.eq(new VarIntWritable(123)), matchFeatureVector(elems));
    EasyMock.replay(ctx);

    ParallelALSFactorizationJob.SolvingReducer reducer = new ParallelALSFactorizationJob.SolvingReducer();
    setField(reducer, "numFeatures", numFeatures);
    setField(reducer, "lambda", lambda);
    setField(reducer, "solver", solver);

    reducer.reduce(new IndexedVarIntWritable(123, 1), Arrays.asList(
        new FeatureVectorWithRatingWritable(456, new Float(ratings.get(0)), col1),
        new FeatureVectorWithRatingWritable(789, new Float(ratings.get(1)), col2)), ctx);

    EasyMock.verify(ctx);
  }

  static FeatureVectorWithRatingWritable matchFeatureVector(final Vector.Element... elements) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof FeatureVectorWithRatingWritable) {
          Vector v = ((FeatureVectorWithRatingWritable) argument).getFeatureVector();
          return MathHelper.consistsOf(v, elements);
        }
        return false;
      }

      @Override
      public void appendTo(StringBuffer buffer) {}
    });
    return null;
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

    File inputFile = getTestTempFile("prefs.txt");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File tmpDir = getTestTempDir("tmp");

    Double na = Double.NaN;
    Matrix preferences = new SparseRowMatrix(new int[] { 4, 4 }, new Vector[] {
        new DenseVector(new double[] {5.0, 5.0, 2.0,  na }),
        new DenseVector(new double[] {2.0,  na, 3.0, 5.0 }),
        new DenseVector(new double[] { na, 5.0,  na, 3.0 }),
        new DenseVector(new double[] {3.0,  na,  na, 5.0 }) });

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
    log.info("Input matrix:\n" + prefsAsText);
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

    Matrix u = MathHelper.readEntries(conf, new Path(outputDir.getAbsolutePath(), "U/part-r-00000"),
        preferences.numRows(), numFeatures);
    Matrix m = MathHelper.readEntries(conf, new Path(outputDir.getAbsolutePath(), "M/part-r-00000"),
      preferences.numCols(), numFeatures);

    RunningAverage avg = new FullRunningAverage();
    sliceIterator = preferences.iterateAll();
    while (sliceIterator.hasNext()) {
      MatrixSlice slice = sliceIterator.next();
      Iterator<Vector.Element> elementIterator = slice.vector().iterateNonZero();
      while (elementIterator.hasNext()) {
        Vector.Element e = elementIterator.next();
        if (!Double.isNaN(e.get())) {
          double pref = e.get();
          double estimate = u.getRow(slice.index()).dot(m.getRow(e.index()));
          double err = pref - estimate;
          avg.addDatum(err * err);
          log.info("Comparing preference of user [" + slice.index() + "] towards item [" + e.index() + "], " +
              "was [" + pref + "] estimate is [" + estimate + ']');
        }
      }
    }
    double rmse = Math.sqrt(avg.getAverage());
    log.info("RMSE: " + rmse);

    assertTrue(rmse < 0.2);
  }

}
