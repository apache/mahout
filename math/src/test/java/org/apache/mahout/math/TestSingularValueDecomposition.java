/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.math;

import java.util.Random;

import org.apache.mahout.math.Algebra;
import org.apache.mahout.math.DenseMatrix;


//To launch this test only : mvn test -Dtest=org.apache.mahout.math.TestSingularValueDecomposition
public class TestSingularValueDecomposition extends MahoutTestCase {
  
  private double[][] testSquare = {
      { 24.0 / 25.0, 43.0 / 25.0 },
      { 57.0 / 25.0, 24.0 / 25.0 }
  };
  
  private double[][] testNonSquare = {
      {  -540.0 / 625.0,  963.0 / 625.0, -216.0 / 625.0 },
      { -1730.0 / 625.0, -744.0 / 625.0, 1008.0 / 625.0 },
      {  -720.0 / 625.0, 1284.0 / 625.0, -288.0 / 625.0 },
      {  -360.0 / 625.0,  192.0 / 625.0, 1756.0 / 625.0 },
  };
  
  private static final double normTolerance = 10e-14;
  
  
  public void testMoreRows() {
    final double[] singularValues = { 123.456, 2.3, 1.001, 0.999 };
    final int rows    = singularValues.length + 2;
    final int columns = singularValues.length;
    Random r = new Random(15338437322523l);
    SingularValueDecomposition svd =
      new SingularValueDecomposition(createTestMatrix(r, rows, columns, singularValues));
    double[] computedSV = svd.getSingularValues();
    assertEquals(singularValues.length, computedSV.length);
    for (int i = 0; i < singularValues.length; ++i) {
      assertEquals(singularValues[i], computedSV[i], 1.0e-10);
    }
  }
  
  
  public void testMoreColumns() {
    final double[] singularValues = { 123.456, 2.3, 1.001, 0.999 };
    final int rows    = singularValues.length;
    final int columns = singularValues.length + 2;
    Random r = new Random(732763225836210l);
    SingularValueDecomposition svd =
      new SingularValueDecomposition(createTestMatrix(r, rows, columns, singularValues));
    double[] computedSV = svd.getSingularValues();
    assertEquals(singularValues.length, computedSV.length);
    for (int i = 0; i < singularValues.length; ++i) {
      assertEquals(singularValues[i], computedSV[i], 1.0e-10);
    }
  }
  
  /** test dimensions */
  public void testDimensions() {
    Matrix matrix = new DenseMatrix(testSquare);
    final int m = matrix.numRows();
    final int n = matrix.numCols();
    SingularValueDecomposition svd = new SingularValueDecomposition(matrix);
    assertEquals(m, svd.getU().numRows());
    assertEquals(m, svd.getU().numCols());
    assertEquals(m, svd.getS().numCols());
    assertEquals(n, svd.getS().numCols());
    assertEquals(n, svd.getV().numRows());
    assertEquals(n, svd.getV().numCols());
    
  }
  
  /** Test based on a dimension 4 Hadamard matrix. */
  // getCovariance to be implemented
  public void testHadamard() {
    Matrix matrix = new DenseMatrix(new double[][] {
        {15.0 / 2.0,  5.0 / 2.0,  9.0 / 2.0,  3.0 / 2.0 },
        { 5.0 / 2.0, 15.0 / 2.0,  3.0 / 2.0,  9.0 / 2.0 },
        { 9.0 / 2.0,  3.0 / 2.0, 15.0 / 2.0,  5.0 / 2.0 },
        { 3.0 / 2.0,  9.0 / 2.0,  5.0 / 2.0, 15.0 / 2.0 }
    });
    SingularValueDecomposition svd = new SingularValueDecomposition(matrix);
    assertEquals(16.0, svd.getSingularValues()[0], 1.0e-14);
    assertEquals( 8.0, svd.getSingularValues()[1], 1.0e-14);
    assertEquals( 4.0, svd.getSingularValues()[2], 1.0e-14);
    assertEquals( 2.0, svd.getSingularValues()[3], 1.0e-14);
    
    Matrix fullCovariance = new DenseMatrix(new double[][] {
        {  85.0 / 1024, -51.0 / 1024, -75.0 / 1024,  45.0 / 1024 },
        { -51.0 / 1024,  85.0 / 1024,  45.0 / 1024, -75.0 / 1024 },
        { -75.0 / 1024,  45.0 / 1024,  85.0 / 1024, -51.0 / 1024 },
        {  45.0 / 1024, -75.0 / 1024, -51.0 / 1024,  85.0 / 1024 }
    });
    
    assertEquals(0.0,Algebra.getNorm(fullCovariance.minus(svd.getCovariance(0.0))),1.0e-14);
    
    
    Matrix halfCovariance = new DenseMatrix(new double[][] {
        {   5.0 / 1024,  -3.0 / 1024,   5.0 / 1024,  -3.0 / 1024 },
        {  -3.0 / 1024,   5.0 / 1024,  -3.0 / 1024,   5.0 / 1024 },
        {   5.0 / 1024,  -3.0 / 1024,   5.0 / 1024,  -3.0 / 1024 },
        {  -3.0 / 1024,   5.0 / 1024,  -3.0 / 1024,   5.0 / 1024 }
    });
    assertEquals(0.0,Algebra.getNorm(halfCovariance.minus(svd.getCovariance(6.0))),1.0e-14);
    
  }
  
  /** test A = USVt */
  public void testAEqualUSVt() {
    checkAEqualUSVt(new DenseMatrix(testSquare));
    checkAEqualUSVt(new DenseMatrix(testNonSquare));
    checkAEqualUSVt(new DenseMatrix(testNonSquare).transpose());
  }
  
  public void checkAEqualUSVt(final Matrix matrix) {
    SingularValueDecomposition svd = new SingularValueDecomposition(matrix);
    Matrix u = svd.getU();
    Matrix s = svd.getS();
    Matrix v = svd.getV();
    
    //pad with 0, to be able to check some properties if some singular values are equal to 0
    if(s.numRows()<matrix.numRows())
    {	
      
      DenseMatrix sp=new DenseMatrix(s.numRows()+1,s.numCols());
      DenseMatrix up=new DenseMatrix(u.numRows(),u.numCols()+1);
      
      
      for(int i=0;i<u.numRows();i++)
        for(int j=0;j<u.numCols();j++)
          up.set(i,j,u.get(i,j));
      
      for(int i=0;i<s.numRows();i++)
        for(int j=0;j<s.numCols();j++)
          sp.set(i,j,s.get(i,j));
      
      u=up;
      s=sp;
    }
    
    double norm = Algebra.getNorm(u.times(s).times(v.transpose()).minus(matrix));
    assertEquals(0, norm, normTolerance);
    
  }
  
  /** test that U is orthogonal */
  public void testUOrthogonal() {
    checkOrthogonal(new SingularValueDecomposition(new DenseMatrix(testSquare)).getU());
    checkOrthogonal(new SingularValueDecomposition(new DenseMatrix(testNonSquare)).getU());
    checkOrthogonal(new SingularValueDecomposition(new DenseMatrix(testNonSquare).transpose()).getU());
  }
  
  /** test that V is orthogonal */
  public void testVOrthogonal() {
    checkOrthogonal(new SingularValueDecomposition(new DenseMatrix(testSquare)).getV());
    checkOrthogonal(new SingularValueDecomposition(new DenseMatrix(testNonSquare)).getV());
    checkOrthogonal(new SingularValueDecomposition(new DenseMatrix(testNonSquare).transpose()).getV());
  }
  
  public void checkOrthogonal(final Matrix m) {
    Matrix mTm = m.transpose().times(m);
    Matrix id  = new DenseMatrix(mTm.numRows(),mTm.numRows());
    for(int i=0;i<mTm.numRows();i++) id.set(i,i,1);
    assertEquals(0, Algebra.getNorm(mTm.minus(id)), normTolerance);
  }
  
  /** test matrices values */
  public void testMatricesValues1() {
    SingularValueDecomposition svd =
      new SingularValueDecomposition(new DenseMatrix(testSquare));
    Matrix uRef = new DenseMatrix(new double[][] {
        { 3.0 / 5.0, 4.0 / 5.0 },
        { 4.0 / 5.0,  -3.0 / 5.0 }
    });
    Matrix sRef = new DenseMatrix(new double[][] {
        { 3.0, 0.0 },
        { 0.0, 1.0 }
    });
    Matrix vRef = new DenseMatrix(new double[][] {
        { 4.0 / 5.0,  -3.0 / 5.0 },
        { 3.0 / 5.0, 4.0 / 5.0 }
    });
    
    // check values against known references
    Matrix u = svd.getU();
    
    assertEquals(0,  Algebra.getNorm(u.minus(uRef)), normTolerance);
    Matrix s = svd.getS();
    assertEquals(0,  Algebra.getNorm(s.minus(sRef)), normTolerance);
    Matrix v = svd.getV();
    assertEquals(0,  Algebra.getNorm(v.minus(vRef)), normTolerance);
  }
  
  
  /** test condition number */
  public void testConditionNumber() {
    SingularValueDecomposition svd =
      new SingularValueDecomposition(new DenseMatrix(testSquare));
    // replace 1.0e-15 with 1.5e-15
    assertEquals(3.0, svd.cond(), 1.5e-15);
  }
  
  private Matrix createTestMatrix(final Random r, final int rows, final int columns,
      final double[] singularValues) {
    final Matrix u = createOrthogonalMatrix(r, rows);
    final Matrix d = createDiagonalMatrix(singularValues, rows, columns);
    final Matrix v = createOrthogonalMatrix(r, columns);
    return u.times(d).times(v);
  }
  
  
  public static Matrix createOrthogonalMatrix(final Random r, final int size) {
    
    final double[][] data = new double[size][size];
    
    for (int i = 0; i < size; ++i) {
      final double[] dataI = data[i];
      double norm2 = 0;
      do {
        
        // generate randomly row I
        for (int j = 0; j < size; ++j) {
          dataI[j] = 2 * r.nextDouble() - 1;
        }
        
        // project the row in the subspace orthogonal to previous rows
        for (int k = 0; k < i; ++k) {
          final double[] dataK = data[k];
          double dotProduct = 0;
          for (int j = 0; j < size; ++j) {
            dotProduct += dataI[j] * dataK[j];
          }
          for (int j = 0; j < size; ++j) {
            dataI[j] -= dotProduct * dataK[j];
          }
        }
        
        // normalize the row
        norm2 = 0;
        for (final double dataIJ : dataI) {
          norm2 += dataIJ * dataIJ;
        }
        final double inv = 1.0 / Math.sqrt(norm2);
        for (int j = 0; j < size; ++j) {
          dataI[j] *= inv;
        }
        
      } while (norm2 * size < 0.01);
    }
    
    return new DenseMatrix(data);
    
  }
  
  public static Matrix createDiagonalMatrix(final double[] diagonal,
      final int rows, final int columns) {
    final double[][] dData = new double[rows][columns];
    for (int i = 0; i < Math.min(rows, columns); ++i) {
      dData[i][i] = diagonal[i];
    }
    return new DenseMatrix(dData);
  }
  
  
}
