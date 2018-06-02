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

package org.apache.mahout.cf.taste.hadoop.similarity.item;

import java.io.BufferedReader;
import java.io.File;
import java.io.FilenameFilter;
import java.util.Arrays;
import java.util.regex.Pattern;

import com.google.common.base.Charsets;
import com.google.common.io.Files;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.CosineSimilarity;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.TanimotoCoefficientSimilarity;
import org.apache.mahout.math.map.OpenIntLongHashMap;
import org.easymock.EasyMock;
import org.junit.Test;

/**
 * Unit tests for the mappers and reducers in org.apache.mahout.cf.taste.hadoop.similarity.item
 * some integration tests with tiny data sets at the end
 */
public final class ItemSimilarityJobTest extends TasteTestCase {

  private static final Pattern TAB = Pattern.compile("\t");

  /**
   * Tests {@link ItemSimilarityJob.MostSimilarItemPairsMapper}
   */
  @Test
  public void testMostSimilarItemsPairsMapper() throws Exception {

    OpenIntLongHashMap indexItemIDMap = new OpenIntLongHashMap();
    indexItemIDMap.put(12, 12L);
    indexItemIDMap.put(34, 34L);
    indexItemIDMap.put(56, 56L);

    Mapper<IntWritable,VectorWritable,EntityEntityWritable,DoubleWritable>.Context context =
      EasyMock.createMock(Mapper.Context.class);

    context.write(new EntityEntityWritable(34L, 56L), new DoubleWritable(0.9));

    EasyMock.replay(context);

    Vector vector = new RandomAccessSparseVector(Integer.MAX_VALUE);
    vector.set(12, 0.2);
    vector.set(56, 0.9);

    ItemSimilarityJob.MostSimilarItemPairsMapper mapper = new ItemSimilarityJob.MostSimilarItemPairsMapper();
    setField(mapper, "indexItemIDMap", indexItemIDMap);
    setField(mapper, "maxSimilarItemsPerItem", 1);

    mapper.map(new IntWritable(34), new VectorWritable(vector), context);

    EasyMock.verify(context);
  }

  /**
   * Tests {@link ItemSimilarityJob.MostSimilarItemPairsReducer}
   */
  @Test
  public void testMostSimilarItemPairsReducer() throws Exception {
    Reducer<EntityEntityWritable,DoubleWritable,EntityEntityWritable,DoubleWritable>.Context context =
      EasyMock.createMock(Reducer.Context.class);

    context.write(new EntityEntityWritable(123L, 456L), new DoubleWritable(0.5));

    EasyMock.replay(context);

    new ItemSimilarityJob.MostSimilarItemPairsReducer().reduce(new EntityEntityWritable(123L, 456L),
        Arrays.asList(new DoubleWritable(0.5), new DoubleWritable(0.5)), context);

    EasyMock.verify(context);
  }

  /**
   * Integration test with a tiny data set
   *
   * <pre>
   * user-item-matrix
   *
   *        Game   Mouse   PC    Disk
   * Jane    -       1      2      -
   * Paul    1       -      1      -
   * Fred    -       -      -      1
   * </pre>
   */
  @Test
  public void testCompleteJob() throws Exception {

    File inputFile = getTestTempFile("prefs.txt");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File tmpDir = getTestTempDir("tmp");

    writeLines(inputFile,
        "2,1,1",
        "1,2,1",
        "3,4,1",
        "1,3,2",
        "2,3,1");

    ItemSimilarityJob similarityJob = new ItemSimilarityJob();

    Configuration conf = getConfiguration();
    conf.set("mapred.input.dir", inputFile.getAbsolutePath());
    conf.set("mapred.output.dir", outputDir.getAbsolutePath());
    conf.setBoolean("mapred.output.compress", false);

    similarityJob.setConf(conf);
    similarityJob.run(new String[] { "--tempDir", tmpDir.getAbsolutePath(), "--similarityClassname",
       CosineSimilarity.class.getName() });
    File outPart = outputDir.listFiles(new FilenameFilter() {
      @Override
      public boolean accept(File dir, String name) {
        return name.startsWith("part-");
      }
    })[0];
    BufferedReader reader = Files.newReader(outPart, Charsets.UTF_8);

    String line;
    int currentLine = 1;
    while ( (line = reader.readLine()) != null) {

      String[] tokens = TAB.split(line);

      long itemAID = Long.parseLong(tokens[0]);
      long itemBID = Long.parseLong(tokens[1]);
      double similarity = Double.parseDouble(tokens[2]);

      if (currentLine == 1) {
        assertEquals(1L, itemAID);
        assertEquals(3L, itemBID);
        assertEquals(0.45, similarity, 0.01);
      }

      if (currentLine == 2) {
        assertEquals(2L, itemAID);
        assertEquals(3L, itemBID);
        assertEquals(0.89, similarity, 0.01);
      }

      currentLine++;
    }

    int linesWritten = currentLine-1;
    assertEquals(2, linesWritten);
  }

  /**
   * integration test for the limitation of the number of computed similarities
   *
   * <pre>
   * user-item-matrix
   *
   *        i1  i2  i3
   *    u1   1   0   1
   *    u2   0   1   1
   *    u3   1   1   0
   *    u4   1   1   1
   *    u5   0   1   0
   *    u6   1   1   0
   *
   *    tanimoto(i1,i2) = 0.5
   *    tanimoto(i2,i3) = 0.333
   *     tanimoto(i3,i1) = 0.4
   *
   *    When we set maxSimilaritiesPerItem to 1 the following pairs should be found:
   *
   *    i1 --> i2
   *    i2 --> i1
   *    i3 --> i1
   * </pre>
   */
  @Test
  public void testMaxSimilaritiesPerItem() throws Exception {

    File inputFile = getTestTempFile("prefsForMaxSimilarities.txt");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File tmpDir = getTestTempDir("tmp");

    writeLines(inputFile,
        "1,1,1",
        "1,3,1",
        "2,2,1",
        "2,3,1",
        "3,1,1",
        "3,2,1",
        "4,1,1",
        "4,2,1",
        "4,3,1",
        "5,2,1",
        "6,1,1",
        "6,2,1");

    ItemSimilarityJob similarityJob =  new ItemSimilarityJob();

    Configuration conf = getConfiguration();
    conf.set("mapred.input.dir", inputFile.getAbsolutePath());
    conf.set("mapred.output.dir", outputDir.getAbsolutePath());
    conf.setBoolean("mapred.output.compress", false);

    similarityJob.setConf(conf);
    similarityJob.run(new String[] { "--tempDir", tmpDir.getAbsolutePath(), "--similarityClassname",
        TanimotoCoefficientSimilarity.class.getName(), "--maxSimilaritiesPerItem", "1" });
    File outPart = outputDir.listFiles(new FilenameFilter() {
      @Override
      public boolean accept(File dir, String name) {
        return name.startsWith("part-");
      }
    })[0];
    BufferedReader reader = Files.newReader(outPart, Charsets.UTF_8);

    String line;
    int currentLine = 1;
    while ((line = reader.readLine()) != null) {

      String[] tokens = TAB.split(line);

      long itemAID = Long.parseLong(tokens[0]);
      long itemBID = Long.parseLong(tokens[1]);
      double similarity = Double.parseDouble(tokens[2]);

      if (currentLine == 1) {
        assertEquals(1L, itemAID);
        assertEquals(2L, itemBID);
        assertEquals(0.5, similarity, 0.0001);
      }

      if (currentLine == 2) {
        assertEquals(1L, itemAID);
        assertEquals(3L, itemBID);
        assertEquals(0.4, similarity, 0.0001);
      }

      currentLine++;
    }

    int linesWritten = currentLine - 1;
    assertEquals(2, linesWritten);
  }

}
