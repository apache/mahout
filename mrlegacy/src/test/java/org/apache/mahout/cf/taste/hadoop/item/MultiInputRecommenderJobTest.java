package org.apache.mahout.cf.taste.hadoop.item;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.TanimotoCoefficientSimilarity;
import org.junit.Test;

public class MultiInputRecommenderJobTest extends TasteTestCase {

  @Test
  public void testMultiInputOnly() throws Exception {

    File inputFile1 = getTestTempFile("prefs1.txt");
    File inputFile2 = getTestTempFile("prefs2.txt");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File similaritiesOutputDir = getTestTempDir("outputSimilarities");
    similaritiesOutputDir.delete();
    File tmpDir = getTestTempDir("tmp");

    writeLines(inputFile1,
        "1,1,5",
        "1,2,5",
        "1,3,2",
        "2,1,2",
        "2,3,3",
        "2,4,5");
    
    writeLines(inputFile2,
        "3,2,5",
        "3,4,3",
        "4,1,3",
        "4,4,5");

    RecommenderJob recommenderJob = new RecommenderJob();

    Configuration conf = getConfiguration();
    recommenderJob.setConf(conf);

    recommenderJob.run(new String[] { 
        "--multiInput", inputFile1.getAbsoluteFile().toString(), inputFile2.getAbsoluteFile().toString(),
        "--output", outputDir.getAbsolutePath(),
        "--tempDir", tmpDir.getAbsolutePath(), 
        "--similarityClassname", TanimotoCoefficientSimilarity.class.getName(), 
        "--numRecommendations", "4",
        "--outputPathForSimilarityMatrix", similaritiesOutputDir.getAbsolutePath() });

    verityRecommendations(outputDir);
    
    veritySimilarities(similaritiesOutputDir);
  }
  
  @Test
  public void testInputOnly() throws Exception {

    File inputFile = getTestTempFile("prefs.txt");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File similaritiesOutputDir = getTestTempDir("outputSimilarities");
    similaritiesOutputDir.delete();
    File tmpDir = getTestTempDir("tmp");

    writeLines(inputFile,
        "1,1,5",
        "1,2,5",
        "1,3,2",
        "2,1,2",
        "2,3,3",
        "2,4,5",
        "3,2,5",
        "3,4,3",
        "4,1,3",
        "4,4,5");

    RecommenderJob recommenderJob = new RecommenderJob();

    Configuration conf = getConfiguration();
    recommenderJob.setConf(conf);

    recommenderJob.run(new String[] { 
        "--input", inputFile.getAbsoluteFile().toString(),
        "--output", outputDir.getAbsolutePath(),
        "--tempDir", tmpDir.getAbsolutePath(), 
        "--similarityClassname", TanimotoCoefficientSimilarity.class.getName(), 
        "--numRecommendations", "4",
        "--outputPathForSimilarityMatrix", similaritiesOutputDir.getAbsolutePath() });

    verityRecommendations(outputDir);
    
    veritySimilarities(similaritiesOutputDir);
  }
  
  @Test
  public void testInputAndMultiInput() throws Exception {

    File inputFile = getTestTempFile("prefs.txt");
    File inputFile1 = getTestTempFile("prefs1.txt");
    File inputFile2 = getTestTempFile("prefs2.txt");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File similaritiesOutputDir = getTestTempDir("outputSimilarities");
    similaritiesOutputDir.delete();
    File tmpDir = getTestTempDir("tmp");

    writeLines(inputFile,
        "1,1,5",
        "1,2,5",
        "1,3,2",
        "2,1,2",
        "2,3,3",
        "2,4,5",
        "3,2,5",
        "3,4,3",
        "4,1,3",
        "4,4,5");
    
    writeLines(inputFile1,
        "1,1,5",
        "1,2,5",
        "1,3,2",
        "2,1,2",
        "2,3,3",
        "2,4,5");
    
    writeLines(inputFile2,
        "3,2,5",
        "3,4,3",
        "4,1,3",
        "4,4,5");

    RecommenderJob recommenderJob = new RecommenderJob();

    Configuration conf = getConfiguration();
    recommenderJob.setConf(conf);

    recommenderJob.run(new String[] { 
        "--input", inputFile.getAbsoluteFile().toString(),
        "--multiInput", inputFile1.getAbsoluteFile().toString(), inputFile2.getAbsoluteFile().toString(),
        "--output", outputDir.getAbsolutePath(),
        "--tempDir", tmpDir.getAbsolutePath(), 
        "--similarityClassname", TanimotoCoefficientSimilarity.class.getName(), 
        "--numRecommendations", "4",
        "--outputPathForSimilarityMatrix", similaritiesOutputDir.getAbsolutePath() });

    verityRecommendations(outputDir);
    
    veritySimilarities(similaritiesOutputDir);
  }
  
  private void verityRecommendations(File output) throws IOException {
    Map<Long,List<RecommendedItem>> recommendations = 
        RecommenderJobTest.readRecommendations(new File(output, "part-r-00000"));
    assertEquals(4, recommendations.size());

    for (Entry<Long,List<RecommendedItem>> entry : recommendations.entrySet()) {
      long userID = entry.getKey();
      List<RecommendedItem> items = entry.getValue();
      assertNotNull(items);
      RecommendedItem item1 = items.get(0);

      if (userID == 1L) {
        assertEquals(1, items.size());
        assertEquals(4L, item1.getItemID());
        assertEquals(4.3, item1.getValue(), 0.05);
      }
      if (userID == 2L) {
        assertEquals(1, items.size());
        assertEquals(2L, item1.getItemID());
        assertEquals(3.3, item1.getValue(), 0.05);
      }
      if (userID == 3L) {
        assertEquals(2, items.size());
        assertEquals(3L, item1.getItemID());
        assertEquals(4.1, item1.getValue(), 0.05);
        RecommendedItem item2 = items.get(1);
        assertEquals(1L, item2.getItemID());
        assertEquals(3.7, item2.getValue(), 0.05);
      }
      if (userID == 4L) {
        assertEquals(2, items.size());
        assertEquals(2L, item1.getItemID());
        assertEquals(4.0, item1.getValue(), 0.05);
        RecommendedItem item2 = items.get(1);
        assertEquals(3L, item2.getItemID());
        assertEquals(3.5, item2.getValue(), 0.05);
      }
    }
  }
  
  private void veritySimilarities(File output) throws IOException {
    Map<Pair<Long, Long>, Double> similarities = 
        RecommenderJobTest.readSimilarities(new File(output, "part-r-00000"));
    assertEquals(6, similarities.size());

    assertEquals(0.25, similarities.get(new Pair<Long, Long>(1L, 2L)), EPSILON);
    assertEquals(0.6666666666666666, similarities.get(new Pair<Long, Long>(1L, 3L)), EPSILON);
    assertEquals(0.5, similarities.get(new Pair<Long, Long>(1L, 4L)), EPSILON);
    assertEquals(0.3333333333333333, similarities.get(new Pair<Long, Long>(2L, 3L)), EPSILON);
    assertEquals(0.25, similarities.get(new Pair<Long, Long>(2L, 4L)), EPSILON);
    assertEquals(0.25, similarities.get(new Pair<Long, Long>(3L, 4L)), EPSILON);
  }
}
