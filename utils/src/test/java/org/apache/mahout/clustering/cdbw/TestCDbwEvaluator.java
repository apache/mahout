package org.apache.mahout.clustering.cdbw;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.clustering.ClusterBase;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.clustering.canopy.CanopyClusteringJob;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.dirichlet.DirichletDriver;
import org.apache.mahout.clustering.dirichlet.models.L1ModelDistribution;
import org.apache.mahout.clustering.fuzzykmeans.FuzzyKMeansDriver;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.TestKmeansClustering;
import org.apache.mahout.clustering.meanshift.MeanShiftCanopyJob;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class TestCDbwEvaluator extends MahoutTestCase {

  public static final double[][] reference = { { 1, 1 }, { 2, 1 }, { 1, 2 }, { 2, 2 }, { 3, 3 }, { 4, 4 }, { 5, 4 }, { 4, 5 },
      { 5, 5 } };

  private List<VectorWritable> sampleData;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    RandomUtils.useTestSeed();
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    // Create testdata directory
    ClusteringTestUtils.rmr("testdata");
    File f = new File("testdata");
    f.mkdir();
    ClusteringTestUtils.rmr("output");
    // Create test data
    sampleData = TestKmeansClustering.getPointsWritable(reference);
    ClusteringTestUtils.writePointsToFile(sampleData, "testdata/file1", fs, conf);
  }

  private void checkRefPoints(int numIterations) throws IOException {
    File out = new File("output");
    assertTrue("output is not Dir", out.isDirectory());
    for (int i = 0; i <= numIterations; i++) {
      out = new File("output/representativePoints-" + i);
      assertTrue("rep-i is not a Dir", out.isDirectory());
      System.out.println(out.getName() + ":");
      File[] files = out.listFiles();
      Configuration conf = new Configuration();
      FileSystem fs = FileSystem.get(conf);
      for (File file : files) {
        if (!file.getName().startsWith(".")) {
          SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(file.getAbsolutePath()), conf);
          try {
            IntWritable clusterId = new IntWritable(0);
            VectorWritable point = new VectorWritable();
            while (reader.next(clusterId, point)) {
              System.out.println("\tC-" + clusterId + ": " + ClusterBase.formatVector(point.get(), null));
            }
          } finally {
            reader.close();
          }
        }
      }
    }
  }

  public void testCanopy() throws Exception { // now run the Job
    CanopyClusteringJob.runJob("testdata", "output", EuclideanDistanceMeasure.class.getName(), 3.1, 2.1);
    int numIterations = 2;
    CDbwDriver.runJob("output/clusters-0", "output/clusteredPoints", "output", EuclideanDistanceMeasure.class.getName(), numIterations, 1);
    checkRefPoints(numIterations);
  }

  public void testKmeans() throws Exception {
    // now run the Canopy job to prime kMeans canopies
    CanopyDriver.runJob("testdata", "output/clusters-0", EuclideanDistanceMeasure.class.getName(), 3.1, 2.1);
    // now run the KMeans job
    KMeansDriver.runJob("testdata", "output/clusters-0", "output", EuclideanDistanceMeasure.class.getName(), 0.001, 10, 1);
    int numIterations = 2;
    CDbwDriver.runJob("output/clusters-2", "output/clusteredPoints", "output", EuclideanDistanceMeasure.class.getName(), numIterations, 1);
    checkRefPoints(numIterations);
  }

  public void testFuzzyKmeans() throws Exception {
    // now run the Canopy job to prime kMeans canopies
    CanopyDriver.runJob("testdata", "output/clusters-0", EuclideanDistanceMeasure.class.getName(), 3.1, 2.1);
    // now run the KMeans job
    FuzzyKMeansDriver.runJob("testdata", "output/clusters-0", "output", EuclideanDistanceMeasure.class.getName(), 0.001, 10, 1, 1, 2);
    int numIterations = 2;
    CDbwDriver.runJob("output/clusters-4", "output/clusteredPoints", "output", EuclideanDistanceMeasure.class.getName(), numIterations, 1);
    checkRefPoints(numIterations);
  }

  public void testMeanShift() throws Exception {
    MeanShiftCanopyJob.runJob("testdata", "output", EuclideanDistanceMeasure.class.getName(), 2.1, 1.0, 0.001, 10);
    int numIterations = 2;
    CDbwDriver.runJob("output/clusters-2", "output/clusteredPoints", "output", EuclideanDistanceMeasure.class.getName(), numIterations, 1);
    checkRefPoints(numIterations);
  }

  public void testDirichlet() throws Exception {
    Vector prototype = new DenseVector(2);
    DirichletDriver.runJob("testdata", "output", L1ModelDistribution.class.getName(), prototype.getClass().getName(), prototype
        .size(), 15, 5, 1.0, 1);
    int numIterations = 2;
    CDbwDriver.runJob("output/clusters-5", "output/clusteredPoints", "output", EuclideanDistanceMeasure.class.getName(), numIterations, 1);
    checkRefPoints(numIterations);
  }

}
