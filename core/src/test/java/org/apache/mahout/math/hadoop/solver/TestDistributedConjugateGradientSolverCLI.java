package org.apache.mahout.math.hadoop.solver;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.hadoop.TestDistributedRowMatrix;
import org.junit.Test;

public class TestDistributedConjugateGradientSolverCLI extends MahoutTestCase
{
  private Vector randomVector(int size, double entryMean) {
    DenseVector v = new DenseVector(size);
    Random r = new Random(1234L);
    
    for (int i = 0; i < size; ++i) {
      v.setQuick(i, r.nextGaussian() * entryMean);
    }
    
    return v;
  }

  private Path saveVector(Configuration conf, Path path, Vector v) throws IOException {
    FileSystem fs = path.getFileSystem(conf);
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path, IntWritable.class, VectorWritable.class);
    
    try {
      writer.append(new IntWritable(0), new VectorWritable(v));
    } finally {
      writer.close();
    }
    return path;
  }
  
  private Vector loadVector(Configuration conf, Path path) throws IOException {
    FileSystem fs = path.getFileSystem(conf);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
    IntWritable key = new IntWritable();
    VectorWritable value = new VectorWritable();
    
    try {
      if (!reader.next(key, value)) {
        throw new IOException("Input vector file is empty.");      
      }
      return value.get();
    } finally {
      reader.close();
    }
  }

  @Test
  public void testSolver() throws Exception {
    Configuration conf = new Configuration();
    Path testData = getTestTempDirPath("testdata");
    DistributedRowMatrix matrix = new TestDistributedRowMatrix().randomDistributedMatrix(
        10, 10, 10, 10, 10.0, true, testData.toString());
    matrix.setConf(conf);
    Path output = getTestTempFilePath("output");
    Path vectorPath = getTestTempFilePath("vector");
    Path tempPath = getTestTempDirPath("tmp");

    Vector vector = randomVector(matrix.numCols(), 10.0);
    saveVector(conf, vectorPath, vector);
        
    String[] args = {
        "-i", matrix.getRowPath().toString(),
        "-o", output.toString(),
        "--tempDir", tempPath.toString(),
        "--vector", vectorPath.toString(),
        "--numRows", "10",
        "--numCols", "10",
        "--symmetric", "true"        
    };
    
    DistributedConjugateGradientSolver solver = new DistributedConjugateGradientSolver();
    solver.job().run(args);
    
    Vector x = loadVector(conf, output);
    
    Vector solvedVector = matrix.times(x);    
    double distance = Math.sqrt(vector.getDistanceSquared(solvedVector));
    assertEquals(0.0, distance, EPSILON);
  }
}
