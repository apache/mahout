package org.apache.mahout.math.hadoop.stochasticsvd;

import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.stochasticsvd.qr.GrammSchmidt;

public class SSVDTestsHelper {

  static void generateDenseInput(Path outputPath,
                                 FileSystem dfs,
                                 Vector svalues,
                                 int m,
                                 int n) throws IOException {
    generateDenseInput(outputPath, dfs, svalues, m, n, 0);
  }

  /**
   * Generate some randome but meaningful input with singular value ratios of n,
   * n-1...1
   * 
   * @param outputPath
   */

  static void generateDenseInput(Path outputPath,
                                 FileSystem dfs,
                                 Vector svalues,
                                 int m,
                                 int n,
                                 int startRowKey) throws IOException {

    Random rnd = new Random();

    int svCnt = svalues.size();
    Matrix v = generateDenseOrthonormalRandom(n, svCnt, rnd);
    Matrix u = generateDenseOrthonormalRandom(m, svCnt, rnd);

    // apply singular values
    Matrix mx = m > n ? v : u;
    for (int i = 0; i < svCnt; i++)
      mx.assignColumn(i, mx.viewColumn(i).times(svalues.getQuick(i)));

    SequenceFile.Writer w =
      SequenceFile.createWriter(dfs,
                                dfs.getConf(),
                                outputPath,
                                IntWritable.class,
                                VectorWritable.class);
    try {

      Vector outV = new DenseVector(n);
      VectorWritable vw = new VectorWritable(outV);
      IntWritable iw = new IntWritable();

      for (int i = 0; i < m; i++) {
        iw.set(startRowKey + i);
        for (int j = 0; j < n; j++)
          outV.setQuick(j, u.viewRow(i).dot(v.viewRow(j)));
        w.append(iw, vw);
      }

    } finally {
      w.close();
    }

  }

  static Matrix generateDenseOrthonormalRandom(int m, int n, Random rnd) {
    DenseMatrix result = new DenseMatrix(m, n);
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < m; i++) {
        result.setQuick(i, j, rnd.nextDouble() - 0.5);
      }
    }
    GrammSchmidt.orthonormalizeColumns(result);
    SSVDPrototypeTest.assertOrthonormality(result, false, 1E-10);
    return result;
  }

  // do not use. for internal consumption only.
  public static void main(String[] args) throws Exception {
    // create 1Gb input for distributed tests.
    Configuration conf = new Configuration();
    FileSystem dfs = FileSystem.getLocal(conf);
    Path outputDir=new Path("/tmp/DRM");
    dfs.mkdirs(outputDir);
    for ( int i = 1; i <= 10; i++ ) {
      generateDenseInput(new Path(outputDir,String.format("part-%05d",i)),dfs,
                         new DenseVector ( new double[] {
                             15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0.8,0.3,0.1,0.01
                         }),1200,10000,(i-1)*1200);
    }
    
  }
}
