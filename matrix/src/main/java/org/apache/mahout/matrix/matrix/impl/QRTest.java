package org.apache.mahout.matrix.matrix.impl;

import org.apache.mahout.matrix.matrix.DoubleFactory2D;
import org.apache.mahout.matrix.matrix.DoubleMatrix2D;
import org.apache.mahout.matrix.matrix.linalg.Algebra;

//import Jama.Matrix.*;
//import Jama.*;

class QRTest {

  /**
   * Constructor for QRTest.
   */
  public QRTest() {
    super();
  }

    public static void main(String args[]) {

       // For COLT
       DoubleMatrix2D xmatrix,ymatrix,zmatrix;

       DoubleFactory2D myfactory;
       myfactory = DoubleFactory2D.dense;
       xmatrix = myfactory.make(8,2);
       ymatrix = myfactory.make(8,1);

       xmatrix.set(0,0,1);
       xmatrix.set(1,0,1);
       xmatrix.set(2,0,1);
       xmatrix.set(3,0,1);
       xmatrix.set(4,0,1);
       xmatrix.set(5,0,1); 
       xmatrix.set(6,0,1);
       xmatrix.set(7,0,1);

       xmatrix.set(0,1,80);
       xmatrix.set(1,1,220);
       xmatrix.set(2,1,140);
       xmatrix.set(3,1,120);
       xmatrix.set(4,1,180);
       xmatrix.set(5,1,100);
       xmatrix.set(6,1,200);
       xmatrix.set(7,1,160);

       ymatrix.set(0,0,0.6);
       ymatrix.set(1,0,6.70);
       ymatrix.set(2,0,5.30);
       ymatrix.set(3,0,4.00);
       ymatrix.set(4,0,6.55);
       ymatrix.set(5,0,2.15);
       ymatrix.set(6,0,6.60);
       ymatrix.set(7,0,5.75);

       Algebra myAlgebra = new Algebra();
       zmatrix = myAlgebra.solve(xmatrix,ymatrix);
       System.err.println(xmatrix);
       System.err.println(ymatrix);
       System.err.println(zmatrix);

       /*
       // For JAMA
       Matrix amatrix,bmatrix,cmatrix;
       amatrix = new Matrix(8,2);
       bmatrix = new Matrix(8,1);

       amatrix.set(0,0,1);
       amatrix.set(1,0,1);
       amatrix.set(2,0,1);
       amatrix.set(3,0,1);
       amatrix.set(4,0,1);
       amatrix.set(5,0,1);
       amatrix.set(6,0,1);
       amatrix.set(7,0,1);

       amatrix.set(0,1,80);
       amatrix.set(1,1,220);
       amatrix.set(2,1,140);
       amatrix.set(3,1,120);
       amatrix.set(4,1,180);
       amatrix.set(5,1,100);
       amatrix.set(6,1,200);
       amatrix.set(7,1,160);

       bmatrix.set(0,0,0.6);
       bmatrix.set(1,0,6.70);
       bmatrix.set(2,0,5.30);
       bmatrix.set(3,0,4.00);
       bmatrix.set(4,0,6.55);
       bmatrix.set(5,0,2.15);
       bmatrix.set(6,0,6.60);
       bmatrix.set(7,0,5.75);

       cmatrix = amatrix.solve(bmatrix);
       amatrix.print(8,5);
       bmatrix.print(8,5);
       cmatrix.print(8,5);
    */
    }


}
