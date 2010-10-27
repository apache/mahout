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

package org.apache.mahout.math.hadoop.similarity;

import java.io.File;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix.MatrixEntryWritable;
import org.apache.mahout.math.hadoop.MathHelper;
import org.apache.mahout.math.hadoop.similarity.RowSimilarityJob.EntriesToVectorsReducer;
import org.apache.mahout.math.hadoop.similarity.RowSimilarityJob.SimilarityReducer;
import org.apache.mahout.math.hadoop.similarity.vector.DistributedTanimotoCoefficientVectorSimilarity;
import org.apache.mahout.math.hadoop.similarity.vector.DistributedVectorSimilarity;
import org.easymock.IArgumentMatcher;
import org.easymock.classextension.EasyMock;
import org.junit.Test;

/**
 * tests {@link RowSimilarityJob}
 */
public final class TestRowSimilarityJob extends MahoutTestCase {

  /**
   * Tests {@link RowSimilarityJob.RowWeightMapper}
   */
  @Test
  public void testRowWeightMapper() throws Exception {
    Mapper<IntWritable,VectorWritable,VarIntWritable,WeightedOccurrence>.Context context =
      EasyMock.createMock(Mapper.Context.class);

    context.write(new VarIntWritable(456), new WeightedOccurrence(123, 0.5, 2.0));
    context.write(new VarIntWritable(789), new WeightedOccurrence(123, 0.1, 2.0));

    EasyMock.replay(context);

    Vector vector = new RandomAccessSparseVector(Integer.MAX_VALUE);
    vector.set(456, 0.5);
    vector.set(789, 0.1);

    RowSimilarityJob.RowWeightMapper mapper = new RowSimilarityJob.RowWeightMapper();
    setField(mapper, "similarity", new DistributedTanimotoCoefficientVectorSimilarity());

    mapper.map(new IntWritable(123), new VectorWritable(vector), context);

    EasyMock.verify(context);
  }

  /**
   * Tests {@link RowSimilarityJob.WeightedOccurrencesPerColumnReducer}
   */
  @Test
  public void testWeightedOccurrencesPerColumnReducer() throws Exception {

    List<WeightedOccurrence> weightedOccurrences = Arrays.asList(new WeightedOccurrence(45, 0.5, 1.0),
        new WeightedOccurrence(78, 3.0, 9.0));

    Reducer<VarIntWritable,WeightedOccurrence,VarIntWritable,WeightedOccurrenceArray>.Context context =
      EasyMock.createMock(Reducer.Context.class);

    context.write(EasyMock.eq(new VarIntWritable(123)), weightedOccurrenceArrayMatches(weightedOccurrences));

    EasyMock.replay(context);

    new RowSimilarityJob.WeightedOccurrencesPerColumnReducer().reduce(new VarIntWritable(123), weightedOccurrences,
        context);

    EasyMock.verify(context);
  }

  /**
   * applies an {@link IArgumentMatcher} to a {@link WeightedOccurrenceArray} that checks whether
   * it matches all {@link WeightedOccurrence}
   */
  static WeightedOccurrenceArray weightedOccurrenceArrayMatches(
      final Collection<WeightedOccurrence> occurrencesToMatch) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof WeightedOccurrenceArray) {
          WeightedOccurrence[] occurrences = ((WeightedOccurrenceArray) argument).getWeightedOccurrences();
          if (occurrences.length != occurrencesToMatch.size()) {
            return false;
          }
          for (WeightedOccurrence occurrence : occurrences) {
            if (!occurrencesToMatch.contains(occurrence)) {
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
   * Tests {@link RowSimilarityJob.CooccurrencesMapper}
   */
  @Test
  public void testCooccurrencesMapper() throws Exception {
    Mapper<VarIntWritable,WeightedOccurrenceArray,WeightedRowPair,Cooccurrence>.Context context =
      EasyMock.createMock(Mapper.Context.class);

    context.write(new WeightedRowPair(34, 34, 1.0, 1.0), new Cooccurrence(12, 0.5, 0.5));
    context.write(new WeightedRowPair(34, 56, 1.0, 3.0), new Cooccurrence(12, 0.5, 1.0));
    context.write(new WeightedRowPair(56, 56, 3.0, 3.0), new Cooccurrence(12, 1.0, 1.0));

    EasyMock.replay(context);

    WeightedOccurrenceArray weightedOccurrences = new WeightedOccurrenceArray(new WeightedOccurrence[] {
        new WeightedOccurrence(34, 0.5, 1.0), new WeightedOccurrence(56, 1.0, 3.0) });

    new RowSimilarityJob.CooccurrencesMapper().map(new VarIntWritable(12), weightedOccurrences, context);

    EasyMock.verify(context);
  }

  /**
   * Tests {@link SimilarityReducer}
   */
  @Test
  public void testSimilarityReducer() throws Exception {

    Reducer<WeightedRowPair,Cooccurrence,SimilarityMatrixEntryKey,MatrixEntryWritable>.Context context =
      EasyMock.createMock(Reducer.Context.class);

    context.write(EasyMock.eq(new SimilarityMatrixEntryKey(12, 0.5)),
        MathHelper.matrixEntryMatches(12, 34, 0.5));
    context.write(EasyMock.eq(new SimilarityMatrixEntryKey(34, 0.5)),
        MathHelper.matrixEntryMatches(34, 12, 0.5));

    EasyMock.replay(context);

    SimilarityReducer reducer = new SimilarityReducer();
    setField(reducer, "similarity", new DistributedTanimotoCoefficientVectorSimilarity());

    reducer.reduce(new WeightedRowPair(12, 34, 3.0, 3.0), Arrays.asList(new Cooccurrence(56, 1.0, 2.0),
        new Cooccurrence(78, 3.0, 6.0)), context);

    EasyMock.verify(context);
  }

  /**
   * Tests {@link SimilarityReducer} in the special case of computing the similarity of a row to
   * itself
   */
  @Test
  public void testSimilarityReducerSelfSimilarity() throws Exception {

    Reducer<WeightedRowPair,Cooccurrence,SimilarityMatrixEntryKey,MatrixEntryWritable>.Context context =
      EasyMock.createMock(Reducer.Context.class);

    context.write(EasyMock.eq(new SimilarityMatrixEntryKey(90, 1.0)), MathHelper.matrixEntryMatches(90, 90, 1.0));

    EasyMock.replay(context);

    SimilarityReducer reducer = new SimilarityReducer();
    setField(reducer, "similarity", new DistributedTanimotoCoefficientVectorSimilarity());

    reducer.reduce(new WeightedRowPair(90, 90, 2.0, 2.0), Arrays.asList(new Cooccurrence(56, 1.0, 2.0),
        new Cooccurrence(78, 3.0, 6.0)), context);

    EasyMock.verify(context);
  }

  /**
   * Tests {@link EntriesToVectorsReducer}
   */
  @Test
  public void testEntriesToVectorsReducer() throws Exception {
    Reducer<SimilarityMatrixEntryKey,MatrixEntryWritable,IntWritable,VectorWritable>.Context context =
      EasyMock.createMock(Reducer.Context.class);

    context.write(EasyMock.eq(new IntWritable(12)), MathHelper.vectorMatches(MathHelper.elem(34, 0.8)));

    EasyMock.replay(context);

    EntriesToVectorsReducer reducer = new EntriesToVectorsReducer();
    setField(reducer, "maxSimilaritiesPerRow", 1);

    reducer.reduce(new SimilarityMatrixEntryKey(12, 1.0), Arrays.asList(
        MathHelper.matrixEntry(12, 34, 0.8),
        MathHelper.matrixEntry(12, 56, 0.7)), context);

    EasyMock.verify(context);

  }

  /**
   * integration test with a tiny data set
   *
   * <pre>
   *
   * input matrix:
   *
   * 1, 0, 1, 1, 0
   * 0, 0, 1, 1, 0
   * 0, 0, 0, 0, 1
   *
   * similarity matrix (via tanimoto):
   *
   * 1,     0.666, 0
   * 0.666, 1,     0
   * 0,     0,     1
   * </pre>
   */
  @Test
  public void testSmallSampleMatrix() throws Exception {

    File inputFile = getTestTempFile("rows");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File tmpDir = getTestTempDir("tmp");

    Configuration conf = new Configuration();
    Path inputPath = new Path(inputFile.getAbsolutePath());
    FileSystem fs = FileSystem.get(inputPath.toUri(), conf);

    MathHelper.writeEntries(new double[][] {
        new double[] { 1, 0, 1, 1, 0 },
        new double[] { 0, 0, 1, 1, 0 },
        new double[] { 0, 0, 0, 0, 1 }},
        fs, conf, inputPath);

    conf.set("mapred.input.dir", inputFile.getAbsolutePath());
    conf.set("mapred.output.dir", outputDir.getAbsolutePath());
    conf.setBoolean("mapred.output.compress", false);

    RowSimilarityJob rowSimilarityJob = new RowSimilarityJob();
    rowSimilarityJob.setConf(conf);

    rowSimilarityJob.run(new String[] { "--numberOfColumns", "3", "--similarityClassname",
        DistributedTanimotoCoefficientVectorSimilarity.class.getName(), "--tempDir", tmpDir.getAbsolutePath() });

    Matrix similarityMatrix =
      MathHelper.readEntries(fs, conf, new Path(outputDir.getAbsolutePath(), "part-r-00000"), 3, 3);
    
    assertNotNull(similarityMatrix);
    assertEquals(3, similarityMatrix.numCols());
    assertEquals(3, similarityMatrix.numRows());

    assertEquals(1.0, similarityMatrix.get(0, 0), EPSILON);
    assertEquals(1.0, similarityMatrix.get(1, 1), EPSILON);
    assertEquals(1.0, similarityMatrix.get(2, 2), EPSILON);

    assertEquals(0.0, similarityMatrix.get(2, 0), EPSILON);
    assertEquals(0.0, similarityMatrix.get(2, 1), EPSILON);
    assertEquals(0.0, similarityMatrix.get(0, 2), EPSILON);
    assertEquals(0.0, similarityMatrix.get(1, 2), EPSILON);

    assertEquals(0.6666, similarityMatrix.get(0, 1), 0.0001);
    assertEquals(0.6666, similarityMatrix.get(1, 0), 0.0001);
  }

  /**
   * a tanimoto-coefficient like {@link DistributedVectorSimilarity} that returns NaN for identical rows
   */
  static class DistributedTanimotoCoefficientExcludeIdentityVectorSimilarity implements DistributedVectorSimilarity {

    private static final DistributedVectorSimilarity tanimoto = new DistributedTanimotoCoefficientVectorSimilarity();

    @Override
    public double similarity(int rowA, int rowB, Iterable<Cooccurrence> cooccurrences, double weightOfVectorA,
        double weightOfVectorB, int numberOfRows) {
      if (rowA == rowB) {
        return Double.NaN;
      }
      return tanimoto.similarity(rowA, rowB, cooccurrences, weightOfVectorA, weightOfVectorB, numberOfRows);
    }

    @Override
    public double weight(Vector v) {
      return tanimoto.weight(v);
    }
  }

  /**
   * integration test for the limitation of the entries of the similarity matrix
   *
   * <pre>
   *      c1 c2 c3 c4 c5 c6
   *  r1  1  0  1  1  0  1
   *  r2  0  1  1  1  1  1
   *  r3  1  1  0  1  0  0
   *
   * tanimoto(r1,r2) = 0.5
   * tanimoto(r2,r3) = 0.333
   * tanimoto(r3,r1) = 0.4
   *
   * When we set maxSimilaritiesPerRow to 1 the following pairs should be found:
   *
   * r1 --> r2
   * r2 --> r1
   * r3 --> r1
   * </pre>
   */
  @Test
  public void testLimitEntriesInSimilarityMatrix() throws Exception {

    File inputFile = getTestTempFile("rows");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File tmpDir = getTestTempDir("tmp");

    Configuration conf = new Configuration();
    Path inputPath = new Path(inputFile.getAbsolutePath());
    FileSystem fs = FileSystem.get(inputPath.toUri(), conf);

    MathHelper.writeEntries(new double[][] {
        new double[] { 1, 0, 1, 1, 0, 1 },
        new double[] { 0, 1, 1, 1, 1, 1 },
        new double[] { 1, 1, 0, 1, 0, 0 }},
        fs, conf, inputPath);

    conf.set("mapred.input.dir", inputFile.getAbsolutePath());
    conf.set("mapred.output.dir", outputDir.getAbsolutePath());
    conf.setBoolean("mapred.output.compress", false);

    RowSimilarityJob rowSimilarityJob = new RowSimilarityJob();
    rowSimilarityJob.setConf(conf);

    rowSimilarityJob.run(new String[] { "--numberOfColumns", "3", "--maxSimilaritiesPerRow", "1",
        "--similarityClassname", DistributedTanimotoCoefficientExcludeIdentityVectorSimilarity.class.getName(),
        "--tempDir", tmpDir.getAbsolutePath() });

    Matrix similarityMatrix =
        MathHelper.readEntries(fs, conf, new Path(outputDir.getAbsolutePath(), "part-r-00000"), 3, 3);
    
    assertNotNull(similarityMatrix);
    assertEquals(3, similarityMatrix.numCols());
    assertEquals(3, similarityMatrix.numRows());

    assertEquals(0.0, similarityMatrix.get(0, 0), EPSILON);
    assertEquals(0.5, similarityMatrix.get(0, 1), EPSILON);
    assertEquals(0.0, similarityMatrix.get(0, 2), EPSILON);

    assertEquals(0.5, similarityMatrix.get(1, 0), EPSILON);
    assertEquals(0.0, similarityMatrix.get(1, 1), EPSILON);
    assertEquals(0.0, similarityMatrix.get(1, 2), EPSILON);

    assertEquals(0.4, similarityMatrix.get(2, 0), EPSILON);
    assertEquals(0.0, similarityMatrix.get(2, 1), EPSILON);
    assertEquals(0.0, similarityMatrix.get(2, 2), EPSILON);
  }

}
