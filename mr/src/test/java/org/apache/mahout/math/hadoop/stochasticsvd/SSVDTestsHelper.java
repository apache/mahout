/*
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

package org.apache.mahout.math.hadoop.stochasticsvd;

import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.stochasticsvd.qr.GramSchmidt;

public final class SSVDTestsHelper {

  private SSVDTestsHelper() {
  }

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

    Random rnd = RandomUtils.getRandom();

    int svCnt = svalues.size();
    Matrix v = generateDenseOrthonormalRandom(n, svCnt, rnd);
    Matrix u = generateDenseOrthonormalRandom(m, svCnt, rnd);

    // apply singular values
    Matrix mx = m > n ? v : u;
    for (int i = 0; i < svCnt; i++) {
      mx.assignColumn(i, mx.viewColumn(i).times(svalues.getQuick(i)));
    }

    SequenceFile.Writer w =
      SequenceFile.createWriter(dfs,
                                dfs.getConf(),
                                outputPath,
                                IntWritable.class,
                                VectorWritable.class);
    try {

      Vector outV = new DenseVector(n);
      Writable vw = new VectorWritable(outV);
      IntWritable iw = new IntWritable();

      for (int i = 0; i < m; i++) {
        iw.set(startRowKey + i);
        for (int j = 0; j < n; j++) {
          outV.setQuick(j, u.viewRow(i).dot(v.viewRow(j)));
        }
        w.append(iw, vw);
      }

    } finally {
      w.close();
    }

  }

  static Matrix generateDenseOrthonormalRandom(int m, int n, Random rnd) {
    Matrix result = new DenseMatrix(m, n);
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < m; i++) {
        result.setQuick(i, j, rnd.nextDouble() - 0.5);
      }
    }
    GramSchmidt.orthonormalizeColumns(result);
    SSVDCommonTest.assertOrthonormality(result, false, 1.0e-10);
    return result;
  }

  // do not use. for internal consumption only.
  public static void main(String[] args) throws Exception {
    // create 1Gb input for distributed tests.
    MahoutTestCase ca = new MahoutTestCase();
    Configuration conf = ca.getConfiguration();
    FileSystem dfs = FileSystem.getLocal(conf);
    Path outputDir=new Path("/tmp/DRM");
    dfs.mkdirs(outputDir);
//    for ( int i = 1; i <= 10; i++ ) {
//      generateDenseInput(new Path(outputDir,String.format("part-%05d",i)),dfs,
//                         new DenseVector ( new double[] {
//                             15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0.8,0.3,0.1,0.01
//                         }),1200,10000,(i-1)*1200);
//    }
    
    /*
     *  create 2Gb sparse 4.5 m x 4.5m input . (similar to wikipedia graph).
     *  
     *  In order to get at 2Gb, we need to generate ~ 40 non-zero items per row average.
     *   
     */
    
    outputDir = new Path("/tmp/DRM-sparse");
    Random rnd = RandomUtils.getRandom();

    SequenceFile.Writer w =
      SequenceFile.createWriter(dfs,
                                dfs.getConf(),
                                new Path(outputDir, "sparse.seq"),
                                IntWritable.class,
                                VectorWritable.class);

    try {

      IntWritable iw = new IntWritable();
      VectorWritable vw = new VectorWritable();
      int avgNZero = 40;
      int n = 4500000;
      for (int i = 1; i < n; i++) {
        Vector vector = new RandomAccessSparseVector(n);
        double nz = Math.round(avgNZero * (rnd.nextGaussian() + 1));
        if (nz < 0) {
          nz = 0;
        }
        for (int j = 1; j < nz; j++) {
          vector.set(rnd.nextInt(n), rnd.nextGaussian() * 25 + 3);
        }
        iw.set(i);
        vw.set(vector);
        w.append(iw, vw);
      }
    } finally {
      w.close();
    }
    
  }
}
