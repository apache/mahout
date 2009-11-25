/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix.matrix.bench;

import org.apache.mahout.matrix.matrix.DoubleFactory2D;
import org.apache.mahout.matrix.matrix.DoubleFactory3D;
import org.apache.mahout.matrix.matrix.DoubleMatrix2D;
import org.apache.mahout.matrix.matrix.DoubleMatrix3D;
/**
Configurable matrix benchmark.
Runs the operations defined in main(args) or in the file specified by args.
To get <a href="doc-files/usage.txt">this overall help</a> on usage type <tt>java org.apache.mahout.matrix.matrix.bench.BenchmarkMatrix -help</tt>.
To get help on usage of a given command, type <tt>java org.apache.mahout.matrix.matrix.bench.BenchmarkMatrix -help &lt;command&gt;</tt>.
Here is the <a href="doc-files/usage_dgemm.txt">help ouput for the dgemm</a> command.
<a href="../doc-files/dgemmColt1.0.1ibm1.3LxPIII_2.txt">Here</a> is a sample result.
For more results see the <a href="../doc-files/performanceLog.html">performance log</a>.
 
@author wolfgang.hoschek@cern.ch
@version 0.5, 10-May-2000
*/
/** 
 * @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported.
 */
@Deprecated
public class BenchmarkMatrix {
/**
 * Benchmark constructor comment.
 */
protected BenchmarkMatrix() {}
/**
 * Not yet documented.
 */
protected static void bench_dgemm(String[] args) {
  String[] types;
  int cpus;
  double minSecs;
  boolean transposeA;
  boolean transposeB;
  double[] densities;
  int[] sizes;
  
  try { // parse
    int k = 1;
    types = new String[] {args[k++]};
    cpus = Integer.parseInt(args[k++]);
    minSecs = new Double(args[k++]).doubleValue();
    densities = new double[] {new Double(args[k++]).doubleValue()};
    transposeA = new Boolean(args[k++]).booleanValue();
    transposeB = new Boolean(args[k++]).booleanValue();
    
    sizes = new int[args.length - k];
    for (int i=0; k<args.length; k++, i++) sizes[i] = Integer.parseInt(args[k]);
  }
  catch (Exception exc) {
    System.out.println(usage(args[0]));
    System.out.println("Ignoring command...\n");
    return;
  }

  org.apache.mahout.matrix.matrix.linalg.SmpBlas.allocateBlas(cpus, org.apache.mahout.matrix.matrix.linalg.SeqBlas.seqBlas);
  Double2DProcedure fun = fun_dgemm(transposeA,transposeB);
  String title = fun.toString();
  String params = transposeA +", "+transposeB+", 1, A, B, 0, C";
  title = title + " dgemm("+params+")";
  run(minSecs,title,fun,types,sizes,densities);
}
/**
 * Not yet documented.
 */
protected static void bench_dgemv(String[] args) {
  String[] types;
  int cpus;
  double minSecs;
  boolean transposeA;
  double[] densities;
  int[] sizes;
  
  try { // parse
    int k = 1;
    types = new String[] {args[k++]};
    cpus = Integer.parseInt(args[k++]);
    minSecs = new Double(args[k++]).doubleValue();
    densities = new double[] {new Double(args[k++]).doubleValue()};
    transposeA = new Boolean(args[k++]).booleanValue();
    
    sizes = new int[args.length - k];
    for (int i=0; k<args.length; k++, i++) sizes[i] = Integer.parseInt(args[k]);
  }
  catch (Exception exc) {
    System.out.println(usage(args[0]));
    System.out.println("Ignoring command...\n");
    return;
  }

  org.apache.mahout.matrix.matrix.linalg.SmpBlas.allocateBlas(cpus, org.apache.mahout.matrix.matrix.linalg.SeqBlas.seqBlas);
  Double2DProcedure fun = fun_dgemv(transposeA);
  String title = fun.toString();
  String params = transposeA +", 1, A, B, 0, C";
  title = title + " dgemv("+params+")";
  run(minSecs,title,fun,types,sizes,densities);
}
/**
 * Not yet documented.
 */
protected static void bench_pow(String[] args) {
  String[] types;
  int cpus;
  double minSecs;
  double[] densities;
  int exponent;
  int[] sizes;
  
  try { // parse
    int k = 1;
    types = new String[] {args[k++]};
    cpus = Integer.parseInt(args[k++]);
    minSecs = new Double(args[k++]).doubleValue();
    densities = new double[] {new Double(args[k++]).doubleValue()};
    exponent = Integer.parseInt(args[k++]);
    
    sizes = new int[args.length - k];
    for (int i=0; k<args.length; k++, i++) sizes[i] = Integer.parseInt(args[k]);
  }
  catch (Exception exc) {
    System.out.println(usage(args[0]));
    System.out.println("Ignoring command...\n");
    return;
  }

  org.apache.mahout.matrix.matrix.linalg.SmpBlas.allocateBlas(cpus, org.apache.mahout.matrix.matrix.linalg.SeqBlas.seqBlas);
  Double2DProcedure fun = fun_pow(exponent);
  String title = fun.toString();
  String params = "A,"+exponent;
  title = title +" pow("+params+")";
  run(minSecs,title,fun,types,sizes,densities);
}
/**
 * Not yet documented.
 */
protected static void benchGeneric(Double2DProcedure fun, String[] args) {
  String[] types;
  int cpus;
  double minSecs;
  double[] densities;
  int[] sizes;
  
  try { // parse
    int k = 1;
    types = new String[] {args[k++]};
    cpus = Integer.parseInt(args[k++]);
    minSecs = new Double(args[k++]).doubleValue();
    densities = new double[] {new Double(args[k++]).doubleValue()};
    
    sizes = new int[args.length - k];
    for (int i=0; k<args.length; k++, i++) sizes[i] = Integer.parseInt(args[k]);
  }
  catch (Exception exc) {
    System.out.println(usage(args[0]));
    System.out.println("Ignoring command...\n");
    return;
  }

  org.apache.mahout.matrix.matrix.linalg.SmpBlas.allocateBlas(cpus, org.apache.mahout.matrix.matrix.linalg.SeqBlas.seqBlas);
  String title = fun.toString();
  run(minSecs,title,fun,types,sizes,densities);
}
/**
 * 
 */
protected static String commands() {
  return "dgemm, dgemv, pow, assign, assignGetSet, assignGetSetQuick, assignLog, assignPlusMult, elementwiseMult, elementwiseMultB, SOR5, SOR8, LUDecompose, LUSolve";
}
/**
 * Linear algebrax matrix-matrix multiply.
 */
protected static Double2DProcedure fun_dgemm(final boolean transposeA, final boolean transposeB) {
  return new Double2DProcedure() {
    public String toString() { return "Blas matrix-matrix mult";  }
    public void setParameters(DoubleMatrix2D G, DoubleMatrix2D H) {
      super.setParameters(G,H);
      D = new org.apache.mahout.matrix.matrix.impl.DenseDoubleMatrix2D(A.rows(),A.columns()).assign(0.5);
      C = D.copy();
      B = D.copy();
    }
    public void init() { C.assign(D); }
    public void apply(org.apache.mahout.matrix.Timer timer) {
      org.apache.mahout.matrix.matrix.linalg.SmpBlas.smpBlas.dgemm(transposeA,transposeB,1,A,B,0,C);
    }
    public double operations() { // Mflops
      double m = A.rows();
      double n = A.columns();
      double p = B.columns();
      return 2.0*m*n*p / 1.0E6; 
    }
  };
}
/**
 * Linear algebrax matrix-matrix multiply.
 */
protected static Double2DProcedure fun_dgemv(final boolean transposeA) {
  return new Double2DProcedure() { 
    public String toString() { return "Blas matrix-vector mult";  }
    public void setParameters(DoubleMatrix2D G, DoubleMatrix2D H) {
      super.setParameters(G,H);
      D = new org.apache.mahout.matrix.matrix.impl.DenseDoubleMatrix2D(A.rows(),A.columns()).assign(0.5);
      C = D.copy();
      B = D.copy();
    }
    public void init() { C.viewRow(0).assign(D.viewRow(0)); }
    public void apply(org.apache.mahout.matrix.Timer timer) {
      org.apache.mahout.matrix.matrix.linalg.SmpBlas.smpBlas.dgemv(transposeA,1,A,B.viewRow(0),0,C.viewRow(0));
    }
    public double operations() { // Mflops
      double m = A.rows();
      double n = A.columns();
      //double p = B.columns();
      return 2.0*m*n / 1.0E6; 
    }
  };
}
/**
 * 2D assign with get,set
 */
protected static Double2DProcedure fun_pow(final int k) {
  return new Double2DProcedure() {
    public double dummy;
    public String toString() { return "matrix to the power of an exponent";  }
    public void setParameters(DoubleMatrix2D A, DoubleMatrix2D B) {
      if (k<0) { // must be nonsingular for inversion
        if (!org.apache.mahout.matrix.matrix.linalg.Property.ZERO.isDiagonallyDominantByRow(A) ||
          !org.apache.mahout.matrix.matrix.linalg.Property.ZERO.isDiagonallyDominantByColumn(A)) {
            org.apache.mahout.matrix.matrix.linalg.Property.ZERO.generateNonSingular(A);
          }
        super.setParameters(A,B);
      }
    }

    public void init() {}    
    public void apply(org.apache.mahout.matrix.Timer timer) {
      org.apache.mahout.matrix.matrix.linalg.Algebra.DEFAULT.pow(A,k);
    }
    public double operations() { // Mflops
      double m = A.rows();
      if (k==0) return m; // identity
      double mflops = 0;
      if (k<0) {
        // LU.decompose
        double N = Math.min(A.rows(),A.columns());
        mflops += (2.0 * N*N*N / 3.0 / 1.0E6);

        // LU.solve
        double n = A.columns();
        double nx = B.columns();
        mflops += (2.0 * nx*(n*n + n) / 1.0E6); 
      }
      // mult
      mflops += 2.0*(Math.abs(k)-1)*m*m*m / 1.0E6;
      return mflops; 
    }
  };
}
/**
 * 2D assign with A.assign(B)
 */
protected static Double2DProcedure funAssign() {
  return new Double2DProcedure() {
    public String toString() { return "A.assign(B) [Mops/sec]";  }
    public void init() { A.assign(0); }    
    public void apply(org.apache.mahout.matrix.Timer timer) {
      A.assign(B);
    }
  };
}
/**
 * 2D assign with get,set
 */
protected static Double2DProcedure funAssignGetSet() {
  return new Double2DProcedure() { 
    public String toString() { return "A.assign(B) via get and set [Mops/sec]";  }
    public void init() { A.assign(0); }    
    public void apply(org.apache.mahout.matrix.Timer timer) {
      int rows=B.rows();
      int columns=B.columns();
      /*
      for (int row=rows; --row >= 0; ) {
        for (int column=columns; --column >= 0; ) {
          A.set(row,column, B.get(row,column));
        }
      }
      */
      for (int row=0; row < rows; row++) {
        for (int column=0; column < columns; column++) {
          A.set(row,column, B.get(row,column));
        }
      }
    }
  };
}
/**
 * 2D assign with getQuick,setQuick
 */
protected static Double2DProcedure funAssignGetSetQuick() {
  return new Double2DProcedure() { 
    public String toString() { return "A.assign(B) via getQuick and setQuick [Mops/sec]";  }
    public void init() { A.assign(0); }    
    public void apply(org.apache.mahout.matrix.Timer timer) {
      int rows=B.rows();
      int columns=B.columns();
      //for (int row=rows; --row >= 0; ) {
      //  for (int column=columns; --column >= 0; ) {
      for (int row=0; row < rows; row++) {
        for (int column=0; column < columns; column++) {
          A.setQuick(row,column, B.getQuick(row,column));
        }
      }
    }
  };
}
/**
 * 2D assign with A.assign(B)
 */
protected static Double2DProcedure funAssignLog() {
  return new Double2DProcedure() { 
    public String toString() { return "A[i,j] = log(A[i,j]) via Blas.assign(fun) [Mflops/sec]";  }
    public void init() { A.assign(C); }    
    public void apply(org.apache.mahout.matrix.Timer timer) {
      org.apache.mahout.matrix.matrix.linalg.SmpBlas.smpBlas.assign(A, org.apache.mahout.jet.math.Functions.log);
    }
  };
}
/**
 * 2D assign with A.assign(B)
 */
protected static Double2DProcedure funAssignPlusMult() {
  return new Double2DProcedure() { 
    public String toString() { return "A[i,j] = A[i,j] + s*B[i,j] via Blas.assign(fun) [Mflops/sec]";  }
    public void init() { A.assign(C); }    
    public void apply(org.apache.mahout.matrix.Timer timer) {
      org.apache.mahout.matrix.matrix.linalg.SmpBlas.smpBlas.assign(A,B, org.apache.mahout.jet.math.Functions.plusMult(0.5));
    }
    public double operations() { // Mflops
      double m = A.rows();
      double n = A.columns();
      return 2*m*n / 1.0E6; 
    }
  };
}
/**
 * Linear algebrax matrix-matrix multiply.
 */
protected static Double2DProcedure funCorrelation() {
  return new Double2DProcedure() { 
    public String toString() { return "xxxxxxx";  }
    public void init() {  }    
    public void setParameters(DoubleMatrix2D A, DoubleMatrix2D B) {
      super.setParameters(A.viewDice(),B); // transposed --> faster (memory aware) iteration in correlation algo
    }
    public void apply(org.apache.mahout.matrix.Timer timer) {
      org.apache.mahout.matrix.matrix.doublealgo.Statistic.correlation(
        org.apache.mahout.matrix.matrix.doublealgo.Statistic.covariance(A));
    }
    public double operations() { // Mflops
      double m = A.rows();
      double n = A.columns();
      return m*(n*n + n) / 1.0E6; 
    }
  };
}
/**
 * Element-by-element matrix-matrix multiply.
 */
protected static Double2DProcedure funElementwiseMult() {
  return new Double2DProcedure() { 
    public String toString() { return "A.assign(F.mult(0.5)) via Blas [Mflops/sec]";  }
    public void init() { A.assign(C); }    
    public void apply(org.apache.mahout.matrix.Timer timer) {
      org.apache.mahout.matrix.matrix.linalg.SmpBlas.smpBlas.assign(A, org.apache.mahout.jet.math.Functions.mult(0.5));
    }
  };
}
/**
 * Element-by-element matrix-matrix multiply.
 */
protected static Double2DProcedure funElementwiseMultB() {
  return new Double2DProcedure() { 
    public String toString() { return "A.assign(B,F.mult) via Blas [Mflops/sec]";  }
    public void init() { A.assign(C); }    
    public void apply(org.apache.mahout.matrix.Timer timer) {
      org.apache.mahout.matrix.matrix.linalg.SmpBlas.smpBlas.assign(A,B, org.apache.mahout.jet.math.Functions.mult);
    }
  };
}
/**
 * 2D assign with get,set
 */
protected static Double2DProcedure funGetQuick() {
  return new Double2DProcedure() {
    public double dummy;
    public String toString() { return "xxxxxxx";  }
    public void init() {}    
    public void apply(org.apache.mahout.matrix.Timer timer) {
      int rows=B.rows();
      int columns=B.columns();
      double sum =0;
      //for (int row=rows; --row >= 0; ) {
      //  for (int column=columns; --column >= 0; ) {
      for (int row=0; row < rows; row++) {
        for (int column=0; column < columns; column++) {
          sum += A.getQuick(row,column);
        }
      }
      dummy = sum;
    }
  };
}
/**
 * 2D assign with getQuick,setQuick
 */
protected static Double2DProcedure funLUDecompose() {
  return new Double2DProcedure() {
    org.apache.mahout.matrix.matrix.linalg.LUDecompositionQuick lu = new org.apache.mahout.matrix.matrix.linalg.LUDecompositionQuick(0);
    public String toString() { return "LU.decompose(A) [Mflops/sec]";  }
    public void init() { A.assign(C); }
    public void apply(org.apache.mahout.matrix.Timer timer) {
      lu.decompose(A);  
    }
    public double operations() { // Mflops
      double N = Math.min(A.rows(),A.columns());
      return (2.0 * N*N*N / 3.0 / 1.0E6); 
    }
  };
}
/**
 * 2D assign with getQuick,setQuick
 */
protected static Double2DProcedure funLUSolve() {
  return new Double2DProcedure() {
    org.apache.mahout.matrix.matrix.linalg.LUDecompositionQuick lu;
    public String toString() { return "LU.solve(A) [Mflops/sec]";  }
    public void setParameters(DoubleMatrix2D A, DoubleMatrix2D B) {
      lu = null;
      if (!org.apache.mahout.matrix.matrix.linalg.Property.ZERO.isDiagonallyDominantByRow(A) ||
        !org.apache.mahout.matrix.matrix.linalg.Property.ZERO.isDiagonallyDominantByColumn(A)) {
          org.apache.mahout.matrix.matrix.linalg.Property.ZERO.generateNonSingular(A);
        }
      super.setParameters(A,B);
      lu = new org.apache.mahout.matrix.matrix.linalg.LUDecompositionQuick(0);
      lu.decompose(A);
    }
    public void init() { B.assign(D); }
    public void apply(org.apache.mahout.matrix.Timer timer) {
      lu.solve(B);  
    }
    public double operations() { // Mflops
      double n = A.columns();
      double nx = B.columns();
      return (2.0 * nx*(n*n + n) / 1.0E6); 
    }
  };
}
/**
 * Linear algebrax matrix-matrix multiply.
 */
protected static Double2DProcedure funMatMultLarge() {
  return new Double2DProcedure() {
    public String toString() { return "xxxxxxx";  }
    public void setParameters(DoubleMatrix2D A, DoubleMatrix2D B) {
      // do not allocate mem for "D" --> safe some mem
      this.A = A;
      this.B = B;
      this.C = A.copy();
    }
    public void init() { C.assign(0); }
    public void apply(org.apache.mahout.matrix.Timer timer) { A.zMult(B,C); }
    public double operations() { // Mflops
      double m = A.rows();
      double n = A.columns();
      double p = B.columns();
      return 2.0*m*n*p / 1.0E6; 
    }
  };
}
/**
 * Linear algebrax matrix-vector multiply.
 */
protected static Double2DProcedure funMatVectorMult() {
  return new Double2DProcedure() { 
    public String toString() { return "xxxxxxx";  }
    public void setParameters(DoubleMatrix2D G, DoubleMatrix2D H) {
      super.setParameters(G,H);
      D = new org.apache.mahout.matrix.matrix.impl.DenseDoubleMatrix2D(A.rows(),A.columns()).assign(0.5);
      C = D.copy();
      B = D.copy();
    }
    public void init() { C.viewRow(0).assign(D.viewRow(0)); }
    public void apply(org.apache.mahout.matrix.Timer timer) { A.zMult(B.viewRow(0),C.viewRow(0)); }
    public double operations() { // Mflops
      double m = A.rows();
      double n = A.columns();
      //double p = B.columns();
      return 2.0*m*n / 1.0E6; 
    }
  };
}
/**
 * 2D assign with get,set
 */
protected static Double2DProcedure funSetQuick() {
  return new Double2DProcedure() {
    private int current;
    private double density;
    public String toString() { return "xxxxxxx";  }
    public void init() { 
      A.assign(0);
      int seed = 123456;
      current = 4*seed+1;
      density = A.cardinality() / (double) A.size();
    }    
    public void apply(org.apache.mahout.matrix.Timer timer) {
      int rows=B.rows();
      int columns=B.columns();
      //for (int row=rows; --row >= 0; ) {
      //  for (int column=columns; --column >= 0; ) {
      for (int row=0; row < rows; row++) {
        for (int column=0; column < columns; column++) {
          // a very fast random number generator (this is an inline version of class org.apache.mahout.jet.random.engine.DRand)
          current *= 0x278DDE6D;
          double random = (double) (current & 0xFFFFFFFFL) * 2.3283064365386963E-10;
          // random uniform in (0.0,1.0)
          if (random < density) 
            A.setQuick(row,column,random);
          else
            A.setQuick(row,column,0);
        }
      }
    }
  };
}
/**
 * 
 */
protected static Double2DProcedure funSOR5() {
  return new Double2DProcedure() {
    double value = 2; 
    double omega = 1.25;
    final double alpha = omega * 0.25;
    final double beta = 1-omega;
    org.apache.mahout.matrix.function.Double9Function function = new org.apache.mahout.matrix.function.Double9Function() {
      public final double apply(  
        double a00, double a01, double a02,
        double a10, double a11, double a12,
        double a20, double a21, double a22) {
        return alpha*a11 + beta*(a01+a10+a12+a21);
      }
    };
    public String toString() { return "A.zAssign8Neighbors(5 point function) [Mflops/sec]";  }
    public void init() { B.assign(D); }
    public void apply(org.apache.mahout.matrix.Timer timer) { A.zAssign8Neighbors(B,function); }
    public double operations() { // Mflops
      double n = A.columns();
      double m = A.rows();
      return 6.0 * m*n / 1.0E6; 
    }
  };
}
/**
 * 
 */
protected static Double2DProcedure funSOR8() {
  return new Double2DProcedure() {
    double value = 2;
    double omega = 1.25;
    final double alpha = omega * 0.25;
    final double beta = 1-omega;
    org.apache.mahout.matrix.function.Double9Function function = new org.apache.mahout.matrix.function.Double9Function() {
      public final double apply(  
        double a00, double a01, double a02,
        double a10, double a11, double a12,
        double a20, double a21, double a22) {
        return alpha*a11 + beta*(a00+a10+a20+a01+a21+a02+a12+a22);
      }
    };
    public String toString() { return "A.zAssign8Neighbors(9 point function) [Mflops/sec]";  }
    public void init() { B.assign(D); }
    public void apply(org.apache.mahout.matrix.Timer timer) { A.zAssign8Neighbors(B,function); }
    public double operations() { // Mflops
      double n = A.columns();
      double m = A.rows();
      return 10.0 * m*n / 1.0E6; 
    }
  };
}
/**
 * 
 */
protected static Double2DProcedure funSort() {
  return new Double2DProcedure() { 
    public String toString() { return "xxxxxxx";  }
    public void init() { A.assign(C); }
    public void apply(org.apache.mahout.matrix.Timer timer) { A.viewSorted(0); }
  };
}
/**
 * Not yet documented.
 */
protected static DoubleFactory2D getFactory(String type) {
  DoubleFactory2D factory;
  if (type.equals("dense")) return DoubleFactory2D.dense;
  if (type.equals("sparse")) return DoubleFactory2D.sparse;
  if (type.equals("rowCompressed")) return DoubleFactory2D.rowCompressed;
  String s = "type="+type+" is unknown. Use one of {dense,sparse,rowCompressed}";
  throw new IllegalArgumentException(s);
}
/**
 * Not yet documented.
 */
protected static Double2DProcedure getGenericFunction(String cmd) {
  if (cmd.equals("dgemm")) return fun_dgemm(false,false);
  else if (cmd.equals("dgemv")) return fun_dgemv(false);
  else if (cmd.equals("pow")) return fun_pow(2);
  else if (cmd.equals("assign")) return funAssign();
  else if (cmd.equals("assignGetSet")) return funAssignGetSet();
  else if (cmd.equals("assignGetSetQuick")) return funAssignGetSetQuick();
  else if (cmd.equals("elementwiseMult")) return funElementwiseMult();
  else if (cmd.equals("elementwiseMultB")) return funElementwiseMultB();
  else if (cmd.equals("SOR5")) return funSOR5();
  else if (cmd.equals("SOR8")) return funSOR8();
  else if (cmd.equals("LUDecompose")) return funLUDecompose();
  else if (cmd.equals("LUSolve")) return funLUSolve();
  else if (cmd.equals("assignLog")) return funAssignLog();
  else if (cmd.equals("assignPlusMult")) return funAssignPlusMult();
  /*
  else if (cmd.equals("xxxxxxxxxxxxxxxxx")) return xxxxx();
  }
  */
  return null;
}
/**
 * Executes a command
 */
protected static boolean handle(String[] params) {
  boolean success = true;
  String cmd = params[0];
  if (cmd.equals("dgemm")) bench_dgemm(params);
  else if (cmd.equals("dgemv")) bench_dgemv(params);
  else if (cmd.equals("pow")) bench_pow(params);
  else {
    Double2DProcedure fun = getGenericFunction(cmd);
    if (fun!=null) {
      benchGeneric(fun,params);
    }
    else {
      success = false;
      String s = "Command="+params[0]+" is illegal or unknown. Should be one of "+commands()+"followed by appropriate parameters.\n"+usage()+"\nIgnoring this line.\n";
      System.out.println(s);
    }
  }        
  return success;
}
/**
 * Runs the matrix benchmark operations defined in args or in the file specified by args0.
 * To get detailed help on usage type java org.apache.mahout.matrix.matrix.bench.BenchmarkMatrix -help
 */
public static void main(String[] args) {
  int n = args.length;
  if (n==0 || (n<=1 && args[0].equals("-help"))) { // overall help
    System.out.println(usage());
    return;
  }
  if (args[0].equals("-help")) { // help on specific command
    if (commands().indexOf(args[1]) < 0) {
      System.out.println(args[1]+": no such command available.\n"+usage());
    }
    else {
      System.out.println(usage(args[1]));
    }
    return;
  }
    
  System.out.println("Colt Matrix benchmark running on\n");
  System.out.println(BenchmarkKernel.systemInfo()+"\n");
    // TODO print out real version info?
  System.out.println("Colt Version is [unknown - now in Mahout]" + "\n");

  org.apache.mahout.matrix.Timer timer = new org.apache.mahout.matrix.Timer().start();
  if (!args[0].equals("-file")) { // interactive mode, commands supplied via java class args
    System.out.println("\n\nExecuting command = "+new org.apache.mahout.matrix.list.ObjectArrayList(args)+" ...");
    handle(args);
  }
  else { // batch mode, read commands from file
    /* 
    parse command file in args[0]
    one command per line (including parameters)
    for example:
    // dgemm dense 2 2.0 false true 0.999 10 30 50 100 250 500 1000
    dgemm dense 2 2.5 false true 0.999 10 50 
    dgemm sparse 2 2.5 false true 0.001 500 1000  
    */
    java.io.BufferedReader reader=null;
    try {
      reader = new java.io.BufferedReader(new java.io.FileReader(args[1]));
    } catch (java.io.IOException exc) { throw new RuntimeException(exc.getMessage()); }
    
    java.io.StreamTokenizer stream = new java.io.StreamTokenizer(reader);
    stream.eolIsSignificant(true);
    stream.slashSlashComments(true); // allow // comments
    stream.slashStarComments(true);  // allow /* comments */
    try {
      org.apache.mahout.matrix.list.ObjectArrayList words = new org.apache.mahout.matrix.list.ObjectArrayList();
      int token;
      while ((token = stream.nextToken()) != stream.TT_EOF) { // while not end of file
        if (token == stream.TT_EOL) { // execute a command line at a time
          //System.out.println(words);
          if (words.size() > 0) { // ignore emty lines
            String[] params = new String[words.size()];
            for (int i=0; i<words.size(); i++) params[i] = (String) words.get(i);

            // execute command
            System.out.println("\n\nExecuting command = "+words+" ...");
            handle(params);
          }
          words.clear();
        }
        else {
          String word;
          org.apache.mahout.matrix.matrix.impl.Former formatter = new org.apache.mahout.matrix.matrix.impl.FormerFactory().create("%G");
          // ok: 2.0 -> 2   wrong: 2.0 -> 2.0 (kills Integer.parseInt())
          if (token == stream.TT_NUMBER) 
            word = formatter.form(stream.nval);
          else 
            word = stream.sval;
          if (word != null) words.add(word);
        }
      }
      reader.close();

      System.out.println("\nCommand file name used: "+args[1]+ "\nTo reproduce and compare results, here it's contents:");
      try {
        reader = new java.io.BufferedReader(new java.io.FileReader(args[1]));
      } catch (java.io.IOException exc) { throw new RuntimeException(exc.getMessage()); }

      /*java.io.InputStream input = new java.io.DataInputStream(new java.io.BufferedInputStream(new java.io.FileInputStream(args[1])));
      BufferedReader d
               = new BufferedReader(new InputStreamReader(in));
               */
      String line;
      while ((line = reader.readLine()) != null) { // while not end of file
        System.out.println(line);
      }
      reader.close();
      
    } catch (java.io.IOException exc) { throw new RuntimeException(exc.getMessage()); }
  }
  
  System.out.println("\nProgram execution took a total of "+timer.minutes() +" minutes.");
  System.out.println("Good bye.");
}
/**
 * Executes procedure repeatadly until more than minSeconds have elapsed.
 */
protected static void run(double minSeconds, String title, Double2DProcedure function, String[] types, int[] sizes, double[] densities) {
  //int[] sizes = {33,500,1000};
  //double[] densities = {0.001,0.01,0.99};
  
  //int[] sizes = {3,5,7,9,30,45,60,61,100,200,300,500,800,1000};
  //double[] densities = {0.001,0.01,0.1,0.999};
  
  //int[] sizes = {3};
  //double[] densities = {0.1};

  DoubleMatrix3D timings = DoubleFactory3D.dense.make(types.length,sizes.length,densities.length);
  org.apache.mahout.matrix.Timer runTime = new org.apache.mahout.matrix.Timer().start();
  for (int k=0; k<types.length; k++) {
    //DoubleFactory2D factory = (k==0 ? DoubleFactory2D.dense : k==1 ? DoubleFactory2D.sparse : DoubleFactory2D.rowCompressed);
    //DoubleFactory2D factory = (k==0 ? DoubleFactory2D.dense : k==1 ? DoubleFactory2D.sparse : k==2 ? DoubleFactory2D.rowCompressed : DoubleFactory2D.rowCompressedModified);
    DoubleFactory2D factory = getFactory(types[k]);
    System.out.print("\n@"); 

    for (int i=0; i<sizes.length; i++) {
      int size = sizes[i];
      System.out.print("x");
      //System.out.println("doing size="+size+"...");

      for (int j=0; j<densities.length; j++) {
        final double density = densities[j];
        System.out.print(".");
        //System.out.println("   doing density="+density+"...");
        float opsPerSec;

        //if (true) {
        //if (!((k==1 && density >= 0.1 && size >=100) || (size>5000 && (k==0 || density>1.0E-4) ))) {
        if (!((k>0 && density >= 0.1 && size >=500) )) {
          double val = 0.5;
          function.A=null; function.B=null; function.C=null; function.D=null; // --> help gc before allocating new mem
          DoubleMatrix2D A = factory.sample(size,size,val,density);
          DoubleMatrix2D B = factory.sample(size,size,val,density);
          function.setParameters(A,B);
          A = null; B = null; // help gc
          double ops = function.operations();
          double secs = BenchmarkKernel.run(minSeconds,function);
          opsPerSec = (float) (ops / secs);
        }
        else { // skip this parameter combination (not used in practice & would take a lot of memory and time)
          opsPerSec = Float.NaN;
        }
        timings.set(k,i,j,opsPerSec);
        //System.out.println(secs);
        //System.out.println(opsPerSec+" Mops/sec\n");
      }
    }
  }
  runTime.stop();
  
  String sliceAxisName = "type";
  String rowAxisName = "size"; 
  String colAxisName = "d"; //"density";
  //String[] sliceNames = {"dense", "sparse"};
  //String[] sliceNames = {"dense", "sparse", "rowCompressed"};
  String[] sliceNames = types;
  //hep.aida.bin.BinFunctions1D F = hep.aida.bin.BinFunctions1D.functions;
  //hep.aida.bin.BinFunction1D[] aggr = null; //{F.mean, F.median, F.sum};
  String[] rowNames = new String[sizes.length];
  String[] colNames = new String[densities.length];
  for (int i=sizes.length; --i >= 0; ) rowNames[i]=Integer.toString(sizes[i]);
  for (int j=densities.length; --j >= 0; ) colNames[j]=Double.toString(densities[j]);
  System.out.println("*");
  // show transposed
  String tmp = rowAxisName; rowAxisName = colAxisName; colAxisName = tmp;
  String[] tmp2 = rowNames; rowNames = colNames; colNames = tmp2;
  timings = timings.viewDice(0,2,1);
  //System.out.println(new org.apache.mahout.matrix.matrix.doublealgo.Formatter("%1.3G").toTitleString(timings,sliceNames,rowNames,colNames,sliceAxisName,rowAxisName,colAxisName,"Performance of "+title,aggr));
  /*
  title = "Speedup of dense over sparse";
  DoubleMatrix2D speedup = org.apache.mahout.matrix.matrix.doublealgo.Transform.div(timings.viewSlice(0).copy(),timings.viewSlice(1));
  System.out.println("\n"+new org.apache.mahout.matrix.matrix.doublealgo.Formatter("%1.3G").toTitleString(speedup,rowNames,colNames,rowAxisName,colAxisName,title,aggr));
  */
  System.out.println("Run took a total of "+runTime+". End of run.");
}
/**
 * Executes procedure repeatadly until more than minSeconds have elapsed.
 */
protected static void runSpecial(double minSeconds, String title, Double2DProcedure function) {
  int[] sizes =        {10000};
  double[] densities = {0.00001};
  boolean[] sparses  = {true};
  
  DoubleMatrix2D timings = DoubleFactory2D.dense.make(sizes.length,4);
  org.apache.mahout.matrix.Timer runTime = new org.apache.mahout.matrix.Timer().start();
  for (int i=0; i<sizes.length; i++) {
    int size = sizes[i];
    double density = densities[i];
    boolean sparse = sparses[i];
    final DoubleFactory2D factory = (sparse ? DoubleFactory2D.sparse : DoubleFactory2D.dense);
    System.out.print("\n@"); 

    System.out.print("x");
    double val = 0.5;
    function.A=null; function.B=null; function.C=null; function.D=null; // --> help gc before allocating new mem
    DoubleMatrix2D A = factory.sample(size,size,val,density);
    DoubleMatrix2D B = factory.sample(size,size,val,density);
    function.setParameters(A,B);
    A = null; B = null; // help gc
    float secs = BenchmarkKernel.run(minSeconds,function);
    double ops = function.operations();
    float opsPerSec = (float) (ops / secs);
    timings.viewRow(i).set(0,sparse ? 0: 1);
    timings.viewRow(i).set(1,size);
    timings.viewRow(i).set(2,density);
    timings.viewRow(i).set(3,opsPerSec);
    //System.out.println(secs);
    //System.out.println(opsPerSec+" Mops/sec\n");
  }
  runTime.stop();
  
  //hep.aida.bin.BinFunctions1D F = hep.aida.bin.BinFunctions1D.functions;
  //hep.aida.bin.BinFunction1D[] aggr = null; //{F.mean, F.median, F.sum};
  String[] rowNames = null;
  String[] colNames = {"dense (y=1,n=0)", "size", "density", "flops/sec"};
  String rowAxisName = null;
  String colAxisName = null;
  System.out.println("*");
  //System.out.println(new org.apache.mahout.matrix.matrix.doublealgo.Formatter("%1.3G").toTitleString(timings,rowNames,colNames,rowAxisName,colAxisName,title,aggr));

  System.out.println("Run took a total of "+runTime+". End of run.");
}
/**
 * Overall usage.
 */
protected static String usage() {
  String usage = 
"\nUsage (help): To get this help, type java org.apache.mahout.matrix.matrix.bench.BenchmarkMatrix -help\n"+
"To get help on a command's args, omit args and type java org.apache.mahout.matrix.matrix.bench.BenchmarkMatrix -help <command>\n" +
"Available commands: "+commands()+"\n\n"+

"Usage (direct): java org.apache.mahout.matrix.matrix.bench.BenchmarkMatrix command {args}\n"+
"Example: dgemm dense 2 2.0 0.999 false true 5 10 25 50 100 250 500\n\n"+

"Usage (batch mode): java org.apache.mahout.matrix.matrix.bench.BenchmarkMatrix -file <file>\nwhere <file> is a text file with each line holding a command followed by appropriate args (comments and empty lines ignored).\n\n"+
"Example file's content:\n" +
"dgemm dense 1 2.0 0.999 false true 5 10 25 50 100 250 500\n"+
"dgemm dense 2 2.0 0.999 false true 5 10 25 50 100 250 500\n\n"+
"/*\n"+
"Java like comments in file are ignored\n"+
"dgemv dense 1 2.0 0.001 false 5 10 25 50 100 250 500 1000\n"+
"dgemv sparse 1 2.0 0.001 false 5 10 25 50 100 250 500 1000\n"+
"dgemv rowCompressed 1 2.0 0.001 false 5 10 25 50 100 250 500 1000\n"+
"*/\n"+
"// more comments ignored\n";
  return usage;
}
/**
 * Usage of a specific command.
 */
protected static String usage(String cmd) {
  String usage = cmd + " description: " + getGenericFunction(cmd).toString() +
  "\nArguments to be supplied:\n" +
  //String usage = "Illegal arguments! Arguments to be supplied:\n" +
    //"\te.g. "+cmd+" dense 2 2.0 false 0.999 10 30 50 100 250 500 1000\n"+
    "\t<operation> <type> <cpus> <minSecs> <density>";
  if (cmd.equals("dgemv")) usage = usage +  " <transposeA>";
  if (cmd.equals("dgemm")) usage = usage +  " <transposeA> <transposeB>";
  if (cmd.equals("pow")) usage = usage +  " <exponent>";
  usage = usage +
    " {sizes}\n" +
    "where\n" +
    "\toperation = the operation to benchmark; in this case: "+cmd+"\n"+
    "\ttype = matrix type to be used; e.g. dense, sparse or rowCompressed\n"+
    "\tcpus = #cpus available; e.g. 1 or 2 or ...\n"+
    "\tminSecs = #seconds each operation shall at least run; e.g. 2.0 is a good number giving realistic timings\n"+
    "\tdensity = the density of the matrices to be benchmarked; e.g. 0.999 is very dense, 0.001 is very sparse\n";
    
  if (cmd.equals("dgemv")) usage = usage +  "\ttransposeA = false or true\n";
  if (cmd.equals("dgemm")) usage = usage +  "\ttransposeA = false or true\n\ttransposeB = false or true\n";
  if (cmd.equals("pow")) usage = usage +  "\texponent = the number of times to multiply; e.g. 1000\n";
  usage = usage +
    "\tsizes = a list of problem sizes; e.g. 100 200 benchmarks squared 100x100 and 200x200 matrices";
  return usage;
}
}
