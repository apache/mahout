/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.doublealgo;

import org.apache.mahout.math.function.BinaryFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.jet.random.engine.MersenneTwister;
import org.apache.mahout.math.jet.random.engine.RandomEngine;
import org.apache.mahout.math.jet.random.sampling.RandomSampler;
import org.apache.mahout.math.matrix.DoubleMatrix1D;
import org.apache.mahout.math.matrix.DoubleMatrix2D;
import org.apache.mahout.math.matrix.DoubleMatrix3D;
import org.apache.mahout.math.matrix.impl.DenseDoubleMatrix2D;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class Statistic {

  /** Euclidean distance function; <tt>Sqrt(Sum( (x[i]-y[i])^2 ))</tt>. */
  public static final VectorVectorFunction EUCLID = new VectorVectorFunction() {
    public double apply(DoubleMatrix1D a, DoubleMatrix1D b) {
      return Math.sqrt(a.aggregate(b, Functions.plus, Functions.chain(Functions.square, Functions.minus)));
    }
  };

  /** Bray-Curtis distance function; <tt>Sum( abs(x[i]-y[i]) )  /  Sum( x[i]+y[i] )</tt>. */
  public static final VectorVectorFunction BRAY_CURTIS = new VectorVectorFunction() {
    public double apply(DoubleMatrix1D a, DoubleMatrix1D b) {
      return a.aggregate(b, Functions.plus, Functions.chain(Functions.abs, Functions.minus)) /
          a.aggregate(b, Functions.plus, Functions.plus);
    }
  };

  /** Canberra distance function; <tt>Sum( abs(x[i]-y[i]) / abs(x[i]+y[i]) )</tt>. */
  public static final VectorVectorFunction CANBERRA = new VectorVectorFunction() {
    private final BinaryFunction fun = new BinaryFunction() {
      public double apply(double a, double b) {
        return Math.abs(a - b) / Math.abs(a + b);
      }
    };

    public double apply(DoubleMatrix1D a, DoubleMatrix1D b) {
      return a.aggregate(b, Functions.plus, fun);
    }
  };

  /** Maximum distance function; <tt>Max( abs(x[i]-y[i]) )</tt>. */
  public static final VectorVectorFunction MAXIMUM = new VectorVectorFunction() {
    public double apply(DoubleMatrix1D a, DoubleMatrix1D b) {
      return a.aggregate(b, Functions.max, Functions.chain(Functions.abs, Functions.minus));
    }
  };

  /** Manhattan distance function; <tt>Sum( abs(x[i]-y[i]) )</tt>. */
  public static final VectorVectorFunction MANHATTAN = new VectorVectorFunction() {
    public double apply(DoubleMatrix1D a, DoubleMatrix1D b) {
      return a.aggregate(b, Functions.plus, Functions.chain(Functions.abs, Functions.minus));
    }
  };


/**
 * Interface that represents a function object: a function that takes 
 * two argument vectors and returns a single value.
 */
  /** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
  @Deprecated
  public interface VectorVectorFunction {

    /**
     * Applies a function to two argument vectors.
     *
     * @param x the first argument vector passed to the function.
     * @param y the second argument vector passed to the function.
     * @return the result of the function.
     */
    double apply(DoubleMatrix1D x, org.apache.mahout.math.matrix.DoubleMatrix1D y);
  }

  /** Makes this class non instantiable, but still let's others inherit from it. */
  private Statistic() {
  }
/**
 * Applies the given aggregation functions to each column and stores the results in a the result matrix.
 * If matrix has shape <tt>m x n</tt>, then result must have shape <tt>aggr.length x n</tt>.
 * Tip: To do aggregations on rows use dice views (transpositions), as in <tt>aggregate(matrix.viewDice(),aggr,result.viewDice())</tt>.
 *
 * @param matrix any matrix; a column holds the values of a given variable.
 * @param aggr the aggregation functions to be applied to each column.
 * @param result the matrix to hold the aggregation results.
 * @return <tt>result</tt> (for convenience only).
 * @see Formatter
 * @see hep.aida.bin.BinFunction1D
 * @see hep.aida.bin.BinFunctions1D
 *
public static DoubleMatrix2D aggregate(DoubleMatrix2D matrix, hep.aida.bin.BinFunction1D[] aggr, DoubleMatrix2D result) {
DynamicBin1D bin = new DynamicBin1D();
double[] elements = new double[matrix.rows()];
org.apache.mahout.math.list.DoubleArrayList values = new DoubleArrayList(elements);
for (int column=matrix.columns(); --column >= 0; ) {
matrix.viewColumn(column).toArray(elements); // copy column into values
bin.clear();
bin.addAllOf(values);
for (int i=aggr.length; --i >= 0; ) {
result.set(i, column, aggr[i].apply(bin));
}
}
return result;
}
 */
/**
 Fills all cell values of the given vector into a bin from which statistics measures can be retrieved efficiently.
 Cells values are copied.
 <br>
 Tip: Use <tt>log.info(bin(vector))</tt> to print most measures computed by the bin. Example:
 <table>
 <td class="PRE">
 <pre>
 Size: 20000
 Sum: 299858.02350278624
 SumOfSquares: 5399184.154095971
 Min: 0.8639113139711261
 Max: 59.75331890541892
 Mean: 14.992901175139313
 RMS: 16.43043540825375
 Variance: 45.17438077634358
 Standard deviation: 6.721188940681818
 Standard error: 0.04752598277592142
 Geometric mean: 13.516615397064466
 Product: Infinity
 Harmonic mean: 11.995174297952191
 Sum of inversions: 1667.337172700724
 Skew: 0.8922838940067878
 Kurtosis: 1.1915828121825598
 Sum of powers(3): 1.1345828465808412E8
 Sum of powers(4): 2.7251055344494686E9
 Sum of powers(5): 7.367125643433887E10
 Sum of powers(6): 2.215370909100143E12
 Moment(0,0): 1.0
 Moment(1,0): 14.992901175139313
 Moment(2,0): 269.95920770479853
 Moment(3,0): 5672.914232904206
 Moment(4,0): 136255.27672247344
 Moment(5,0): 3683562.8217169433
 Moment(6,0): 1.1076854545500715E8
 Moment(0,mean()): 1.0
 Moment(1,mean()): -2.0806734113421045E-14
 Moment(2,mean()): 45.172122057305664
 Moment(3,mean()): 270.92018671421
 Moment(4,mean()): 8553.8664869067
 Moment(5,mean()): 153357.41712233616
 Moment(6,mean()): 4273757.570142922
 25%, 50% and 75% Quantiles: 10.030074811938091, 13.977982089912224,
 18.86124362967137
 quantileInverse(mean): 0.559163335012079
 Distinct elements & frequencies not printed (too many).
 </pre>
 </td>
 </table>

 @param vector the vector to analyze.
 @return a bin holding the statistics measures of the vector.
  *
 public static DynamicBin1D bin(DoubleMatrix1D vector) {
 DynamicBin1D bin = new DynamicBin1D();
 bin.addAllOf(DoubleFactory1D.dense.toList(vector));
 return bin;
 }
 */
  /**
   * Modifies the given covariance matrix to be a correlation matrix (in-place). The correlation matrix is a square,
   * symmetric matrix consisting of nothing but correlation coefficients. The rows and the columns represent the
   * variables, the cells represent correlation coefficients. The diagonal cells (i.e. the correlation between a
   * variable and itself) will equal 1, for the simple reason that the correlation coefficient of a variable with itself
   * equals 1. The correlation of two column vectors x and y is given by <tt>corr(x,y) = cov(x,y) /
   * (stdDev(x)*stdDev(y))</tt> (Pearson's correlation coefficient). A correlation coefficient varies between -1 (for a
   * perfect negative relationship) to +1 (for a perfect positive relationship). See the <A
   * HREF="http://www.cquest.utoronto.ca/geog/ggr270y/notes/not05efg.html"> math definition</A> and <A
   * HREF="http://www.stat.berkeley.edu/users/stark/SticiGui/Text/gloss.htm#correlation_coef"> another def</A>. Compares
   * two column vectors at a time. Use dice views to compare two row vectors at a time.
   *
   * @param covariance a covariance matrix, as, for example, returned by method {@link #covariance(DoubleMatrix2D)}.
   * @return the modified covariance, now correlation matrix (for convenience only).
   */
  public static DoubleMatrix2D correlation(DoubleMatrix2D covariance) {
    for (int i = covariance.columns(); --i >= 0;) {
      for (int j = i; --j >= 0;) {
        double stdDev1 = Math.sqrt(covariance.getQuick(i, i));
        double stdDev2 = Math.sqrt(covariance.getQuick(j, j));
        double cov = covariance.getQuick(i, j);
        double corr = cov / (stdDev1 * stdDev2);

        covariance.setQuick(i, j, corr);
        covariance.setQuick(j, i, corr); // symmetric
      }
    }
    for (int i = covariance.columns(); --i >= 0;) {
      covariance.setQuick(i, i, 1);
    }

    return covariance;
  }

  /**
   * Constructs and returns the covariance matrix of the given matrix. The covariance matrix is a square, symmetric
   * matrix consisting of nothing but covariance coefficients. The rows and the columns represent the variables, the
   * cells represent covariance coefficients. The diagonal cells (i.e. the covariance between a variable and itself)
   * will equal the variances. The covariance of two column vectors x and y is given by <tt>cov(x,y) = (1/n) *
   * Sum((x[i]-mean(x)) * (y[i]-mean(y)))</tt>. See the <A HREF="http://www.cquest.utoronto.ca/geog/ggr270y/notes/not05efg.html">
   * math definition</A>. Compares two column vectors at a time. Use dice views to compare two row vectors at a time.
   *
   * @param matrix any matrix; a column holds the values of a given variable.
   * @return the covariance matrix (<tt>n x n, n=matrix.columns</tt>).
   */
  public static DoubleMatrix2D covariance(DoubleMatrix2D matrix) {
    int rows = matrix.rows();
    int columns = matrix.columns();
    DoubleMatrix2D covariance = new DenseDoubleMatrix2D(columns, columns);

    double[] sums = new double[columns];
    DoubleMatrix1D[] cols = new DoubleMatrix1D[columns];
    for (int i = columns; --i >= 0;) {
      cols[i] = matrix.viewColumn(i);
      sums[i] = cols[i].zSum();
    }

    for (int i = columns; --i >= 0;) {
      for (int j = i + 1; --j >= 0;) {
        double sumOfProducts = cols[i].zDotProduct(cols[j]);
        double cov = (sumOfProducts - sums[i] * sums[j] / rows) / rows;
        covariance.setQuick(i, j, cov);
        covariance.setQuick(j, i, cov); // symmetric
      }
    }
    return covariance;
  }
/**
 2-d OLAP cube operator; Fills all cells of the given vectors into the given histogram.
 If you use hep.aida.ref.Converter.toString(histo) on the result, the OLAP cube of x-"column" vs. y-"column" , summing the weights "column" will be printed.
 For example, aggregate sales by product by region.
 <p>
 Computes the distinct values of x and y, yielding histogram axes that capture one distinct value per bin.
 Then fills the histogram.
 <p>
 Example output:
 <table>
 <td class="PRE">
 <pre>
 Cube:
 &nbsp;&nbsp;&nbsp;Entries=5000, ExtraEntries=0
 &nbsp;&nbsp;&nbsp;MeanX=4.9838, RmsX=NaN
 &nbsp;&nbsp;&nbsp;MeanY=2.5304, RmsY=NaN
 &nbsp;&nbsp;&nbsp;xAxis: Min=0, Max=10, Bins=11
 &nbsp;&nbsp;&nbsp;yAxis: Min=0, Max=5, Bins=6
 Heights:
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| X
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| 0   1   2   3   4   5   6   7   8   9   10  | Sum
 ----------------------------------------------------------
 Y 5   |  30  53  51  52  57  39  65  61  55  49  22 |  534
 &nbsp;&nbsp;4   |  43 106 112  96  92  94 107  98  98 110  47 | 1003
 &nbsp;&nbsp;3   |  39 134  87  93 102 103 110  90 114  98  51 | 1021
 &nbsp;&nbsp;2   |  44  81 113  96 101  86 109  83 111  93  42 |  959
 &nbsp;&nbsp;1   |  54  94 103  99 115  92  98  97 103  90  44 |  989
 &nbsp;&nbsp;0   |  24  54  52  44  42  56  46  47  56  53  20 |  494
 ----------------------------------------------------------
 &nbsp;&nbsp;Sum | 234 522 518 480 509 470 535 476 537 493 226 |
 </pre>
 </td>
 </table>
 @return the histogram containing the cube.
 @throws IllegalArgumentException if <tt>x.size() != y.size() || y.size() != weights.size()</tt>.
  *
 public static hep.aida.IHistogram2D cube(DoubleMatrix1D x, DoubleMatrix1D y, DoubleMatrix1D weights) {
 if (x.size() != y.size() || y.size() != weights.size()) throw new IllegalArgumentException("vectors must have same size");

 double epsilon = 1.0E-9;
 org.apache.mahout.math.list.DoubleArrayList distinct = new DoubleArrayList();
 double[] vals = new double[x.size()];
 org.apache.mahout.math.list.DoubleArrayList sorted = new DoubleArrayList(vals);

 // compute distinct values of x
 x.toArray(vals); // copy x into vals
 sorted.sort();
 org.apache.mahout.math.jet.stat.Descriptive.frequencies(sorted, distinct, null);
 // since bins are right-open [from,to) we need an additional dummy bin so that the last distinct value does not fall into the overflow bin
 if (distinct.size()>0) distinct.add(distinct.get(distinct.size()-1) + epsilon);
 distinct.trimToSize();
 hep.aida.IAxis xaxis = new hep.aida.ref.VariableAxis(distinct.elements());

 // compute distinct values of y
 y.toArray(vals);
 sorted.sort();
 org.apache.mahout.math.jet.stat.Descriptive.frequencies(sorted, distinct, null);
 // since bins are right-open [from,to) we need an additional dummy bin so that the last distinct value does not fall into the overflow bin
 if (distinct.size()>0) distinct.add(distinct.get(distinct.size()-1) + epsilon);
 distinct.trimToSize();
 hep.aida.IAxis yaxis = new hep.aida.ref.VariableAxis(distinct.elements());

 hep.aida.IHistogram2D histo = new hep.aida.ref.Histogram2D("Cube",xaxis,yaxis);
 return histogram(histo,x,y,weights);
 }
 */
/**
 3-d OLAP cube operator; Fills all cells of the given vectors into the given histogram.
 If you use hep.aida.ref.Converter.toString(histo) on the result, the OLAP cube of x-"column" vs. y-"column" vs. z-"column", summing the weights "column" will be printed.
 For example, aggregate sales by product by region by time.
 <p>
 Computes the distinct values of x and y and z, yielding histogram axes that capture one distinct value per bin.
 Then fills the histogram.
 @return the histogram containing the cube.
 @throws IllegalArgumentException if <tt>x.size() != y.size() || x.size() != z.size() || x.size() != weights.size()</tt>.
  *
 public static hep.aida.IHistogram3D cube(DoubleMatrix1D x, DoubleMatrix1D y, DoubleMatrix1D z, DoubleMatrix1D weights) {
 if (x.size() != y.size() || x.size() != z.size() || x.size() != weights.size()) throw new IllegalArgumentException("vectors must have same size");

 double epsilon = 1.0E-9;
 org.apache.mahout.math.list.DoubleArrayList distinct = new DoubleArrayList();
 double[] vals = new double[x.size()];
 org.apache.mahout.math.list.DoubleArrayList sorted = new DoubleArrayList(vals);

 // compute distinct values of x
 x.toArray(vals); // copy x into vals
 sorted.sort();
 org.apache.mahout.math.jet.stat.Descriptive.frequencies(sorted, distinct, null);
 // since bins are right-open [from,to) we need an additional dummy bin so that the last distinct value does not fall into the overflow bin
 if (distinct.size()>0) distinct.add(distinct.get(distinct.size()-1) + epsilon);
 distinct.trimToSize();
 hep.aida.IAxis xaxis = new hep.aida.ref.VariableAxis(distinct.elements());

 // compute distinct values of y
 y.toArray(vals);
 sorted.sort();
 org.apache.mahout.math.jet.stat.Descriptive.frequencies(sorted, distinct, null);
 // since bins are right-open [from,to) we need an additional dummy bin so that the last distinct value does not fall into the overflow bin
 if (distinct.size()>0) distinct.add(distinct.get(distinct.size()-1) + epsilon);
 distinct.trimToSize();
 hep.aida.IAxis yaxis = new hep.aida.ref.VariableAxis(distinct.elements());

 // compute distinct values of z
 z.toArray(vals);
 sorted.sort();
 org.apache.mahout.math.jet.stat.Descriptive.frequencies(sorted, distinct, null);
 // since bins are right-open [from,to) we need an additional dummy bin so that the last distinct value does not fall into the overflow bin
 if (distinct.size()>0) distinct.add(distinct.get(distinct.size()-1) + epsilon);
 distinct.trimToSize();
 hep.aida.IAxis zaxis = new hep.aida.ref.VariableAxis(distinct.elements());

 hep.aida.IHistogram3D histo = new hep.aida.ref.Histogram3D("Cube",xaxis,yaxis,zaxis);
 return histogram(histo,x,y,z,weights);
 }
 */


  /**
   * Constructs and returns the distance matrix of the given matrix. The distance matrix is a square, symmetric matrix
   * consisting of nothing but distance coefficients. The rows and the columns represent the variables, the cells
   * represent distance coefficients. The diagonal cells (i.e. the distance between a variable and itself) will be zero.
   * Compares two column vectors at a time. Use dice views to compare two row vectors at a time.
   *
   * @param matrix           any matrix; a column holds the values of a given variable (vector).
   * @param distanceFunction (EUCLID, CANBERRA, ..., or any user defined distance function operating on two vectors).
   * @return the distance matrix (<tt>n x n, n=matrix.columns</tt>).
   */
  public static DoubleMatrix2D distance(DoubleMatrix2D matrix, VectorVectorFunction distanceFunction) {
    int columns = matrix.columns();
    DoubleMatrix2D distance = new DenseDoubleMatrix2D(columns, columns);

    // cache views
    DoubleMatrix1D[] cols = new DoubleMatrix1D[columns];
    for (int i = columns; --i >= 0;) {
      cols[i] = matrix.viewColumn(i);
    }

    // work out all permutations
    for (int i = columns; --i >= 0;) {
      for (int j = i; --j >= 0;) {
        double d = distanceFunction.apply(cols[i], cols[j]);
        distance.setQuick(i, j, d);
        distance.setQuick(j, i, d); // symmetric
      }
    }
    return distance;
  }


  /**
   * Constructs and returns a sampling view with a size of <tt>round(matrix.size() * fraction)</tt>. Samples "without
   * replacement" from the uniform distribution.
   *
   * @param matrix          any matrix.
   * @param fraction        the percentage of rows to be included in the view.
   * @param randomGenerator a uniform random number generator; set this parameter to <tt>null</tt> to use a default
   *                        generator seeded with the current time.
   * @return the sampling view.
   * @throws IllegalArgumentException if <tt>! (0 <= rowFraction <= 1 && 0 <= columnFraction <= 1)</tt>.
   * @see org.apache.mahout.math.jet.random.sampling.RandomSampler
   */
  public static DoubleMatrix1D viewSample(DoubleMatrix1D matrix, double fraction, RandomEngine randomGenerator) {
    // check preconditions and allow for a little tolerance
    double epsilon = 1.0e-09;
    if (fraction < 0 - epsilon || fraction > 1 + epsilon) {
      throw new IllegalArgumentException();
    }
    if (fraction < 0) {
      fraction = 0;
    }
    if (fraction > 1) {
      fraction = 1;
    }

    // random generator seeded with current time
    if (randomGenerator == null) {
      randomGenerator = new MersenneTwister((int) System.currentTimeMillis());
    }

    int ncols = (int) Math.round(matrix.size() * fraction);
    int max = ncols;
    long[] selected = new long[max]; // sampler works on long's, not int's

    // sample
    int n = ncols;
    int N = matrix.size();
    RandomSampler.sample(n, N, n, 0, selected, 0, randomGenerator);
    int[] selectedCols = new int[n];
    for (int i = 0; i < n; i++) {
      selectedCols[i] = (int) selected[i];
    }

    return matrix.viewSelection(selectedCols);
  }

  /**
   * Constructs and returns a sampling view with <tt>round(matrix.rows() * rowFraction)</tt> rows and
   * <tt>round(matrix.columns() * columnFraction)</tt> columns. Samples "without replacement". Rows and columns are
   * randomly chosen from the uniform distribution. Examples: <table border="1" cellspacing="0"> <tr valign="top"
   * align="center"> <td> <div align="left"><tt>matrix</tt></div> </td> <td> <div align="left"><tt>rowFraction=0.2<br>
   * columnFraction=0.2</tt></div> </td> <td> <div align="left"><tt>rowFraction=0.2<br> columnFraction=1.0 </tt></div>
   * </td> <td> <div align="left"><tt>rowFraction=1.0<br> columnFraction=0.2 </tt></div> </td> </tr> <tr valign="top">
   * <td><tt> 10&nbsp;x&nbsp;10&nbsp;matrix<br> &nbsp;1&nbsp;&nbsp;2&nbsp;&nbsp;3&nbsp;&nbsp;4&nbsp;&nbsp;5&nbsp;&nbsp;6&nbsp;&nbsp;7&nbsp;&nbsp;8&nbsp;&nbsp;9&nbsp;&nbsp;10<br>
   * 11&nbsp;12&nbsp;13&nbsp;14&nbsp;15&nbsp;16&nbsp;17&nbsp;18&nbsp;19&nbsp;&nbsp;20<br>
   * 21&nbsp;22&nbsp;23&nbsp;24&nbsp;25&nbsp;26&nbsp;27&nbsp;28&nbsp;29&nbsp;&nbsp;30<br>
   * 31&nbsp;32&nbsp;33&nbsp;34&nbsp;35&nbsp;36&nbsp;37&nbsp;38&nbsp;39&nbsp;&nbsp;40<br>
   * 41&nbsp;42&nbsp;43&nbsp;44&nbsp;45&nbsp;46&nbsp;47&nbsp;48&nbsp;49&nbsp;&nbsp;50<br>
   * 51&nbsp;52&nbsp;53&nbsp;54&nbsp;55&nbsp;56&nbsp;57&nbsp;58&nbsp;59&nbsp;&nbsp;60<br>
   * 61&nbsp;62&nbsp;63&nbsp;64&nbsp;65&nbsp;66&nbsp;67&nbsp;68&nbsp;69&nbsp;&nbsp;70<br>
   * 71&nbsp;72&nbsp;73&nbsp;74&nbsp;75&nbsp;76&nbsp;77&nbsp;78&nbsp;79&nbsp;&nbsp;80<br>
   * 81&nbsp;82&nbsp;83&nbsp;84&nbsp;85&nbsp;86&nbsp;87&nbsp;88&nbsp;89&nbsp;&nbsp;90<br>
   * 91&nbsp;92&nbsp;93&nbsp;94&nbsp;95&nbsp;96&nbsp;97&nbsp;98&nbsp;99&nbsp;100 </tt> </td> <td><tt>
   * 2&nbsp;x&nbsp;2&nbsp;matrix<br> 43&nbsp;50<br> 53&nbsp;60 </tt></td> <td><tt> 2&nbsp;x&nbsp;10&nbsp;matrix<br>
   * 41&nbsp;42&nbsp;43&nbsp;44&nbsp;45&nbsp;46&nbsp;47&nbsp;48&nbsp;49&nbsp;&nbsp;50<br>
   * 91&nbsp;92&nbsp;93&nbsp;94&nbsp;95&nbsp;96&nbsp;97&nbsp;98&nbsp;99&nbsp;100 </tt> </td> <td><tt>
   * 10&nbsp;x&nbsp;2&nbsp;matrix<br> &nbsp;4&nbsp;&nbsp;8<br> 14&nbsp;18<br> 24&nbsp;28<br> 34&nbsp;38<br>
   * 44&nbsp;48<br> 54&nbsp;58<br> 64&nbsp;68<br> 74&nbsp;78<br> 84&nbsp;88<br> 94&nbsp;98 </tt> </td> </tr> </table>
   *
   * @param matrix          any matrix.
   * @param rowFraction     the percentage of rows to be included in the view.
   * @param columnFraction  the percentage of columns to be included in the view.
   * @param randomGenerator a uniform random number generator; set this parameter to <tt>null</tt> to use a default
   *                        generator seeded with the current time.
   * @return the sampling view.
   * @throws IllegalArgumentException if <tt>! (0 <= rowFraction <= 1 && 0 <= columnFraction <= 1)</tt>.
   * @see org.apache.mahout.math.jet.random.sampling.RandomSampler
   */
  public static DoubleMatrix2D viewSample(DoubleMatrix2D matrix, double rowFraction, double columnFraction,
                                          RandomEngine randomGenerator) {
    // check preconditions and allow for a little tolerance
    double epsilon = 1.0e-09;
    if (rowFraction < 0 - epsilon || rowFraction > 1 + epsilon) {
      throw new IllegalArgumentException();
    }
    if (rowFraction < 0) {
      rowFraction = 0;
    }
    if (rowFraction > 1) {
      rowFraction = 1;
    }

    if (columnFraction < 0 - epsilon || columnFraction > 1 + epsilon) {
      throw new IllegalArgumentException();
    }
    if (columnFraction < 0) {
      columnFraction = 0;
    }
    if (columnFraction > 1) {
      columnFraction = 1;
    }

    // random generator seeded with current time
    if (randomGenerator == null) {
      randomGenerator = new MersenneTwister((int) System.currentTimeMillis());
    }

    int nrows = (int) Math.round(matrix.rows() * rowFraction);
    int ncols = (int) Math.round(matrix.columns() * columnFraction);
    int max = Math.max(nrows, ncols);
    long[] selected = new long[max]; // sampler works on long's, not int's

    // sample rows
    int n = nrows;
    int N = matrix.rows();
    RandomSampler.sample(n, N, n, 0, selected, 0, randomGenerator);
    int[] selectedRows = new int[n];
    for (int i = 0; i < n; i++) {
      selectedRows[i] = (int) selected[i];
    }

    // sample columns
    n = ncols;
    N = matrix.columns();
    RandomSampler.sample(n, N, n, 0, selected, 0, randomGenerator);
    int[] selectedCols = new int[n];
    for (int i = 0; i < n; i++) {
      selectedCols[i] = (int) selected[i];
    }

    return matrix.viewSelection(selectedRows, selectedCols);
  }

  /**
   * Constructs and returns a sampling view with <tt>round(matrix.slices() * sliceFraction)</tt> slices and
   * <tt>round(matrix.rows() * rowFraction)</tt> rows and <tt>round(matrix.columns() * columnFraction)</tt> columns.
   * Samples "without replacement". Slices, rows and columns are randomly chosen from the uniform distribution.
   *
   * @param matrix          any matrix.
   * @param sliceFraction   the percentage of slices to be included in the view.
   * @param rowFraction     the percentage of rows to be included in the view.
   * @param columnFraction  the percentage of columns to be included in the view.
   * @param randomGenerator a uniform random number generator; set this parameter to <tt>null</tt> to use a default
   *                        generator seeded with the current time.
   * @return the sampling view.
   * @throws IllegalArgumentException if <tt>! (0 <= sliceFraction <= 1 && 0 <= rowFraction <= 1 && 0 <= columnFraction
   *                                  <= 1)</tt>.
   * @see org.apache.mahout.math.jet.random.sampling.RandomSampler
   */
  public static DoubleMatrix3D viewSample(DoubleMatrix3D matrix, double sliceFraction, double rowFraction,
                                          double columnFraction, RandomEngine randomGenerator) {
    // check preconditions and allow for a little tolerance
    double epsilon = 1.0e-09;
    if (sliceFraction < 0 - epsilon || sliceFraction > 1 + epsilon) {
      throw new IllegalArgumentException();
    }
    if (sliceFraction < 0) {
      sliceFraction = 0;
    }
    if (sliceFraction > 1) {
      sliceFraction = 1;
    }

    if (rowFraction < 0 - epsilon || rowFraction > 1 + epsilon) {
      throw new IllegalArgumentException();
    }
    if (rowFraction < 0) {
      rowFraction = 0;
    }
    if (rowFraction > 1) {
      rowFraction = 1;
    }

    if (columnFraction < 0 - epsilon || columnFraction > 1 + epsilon) {
      throw new IllegalArgumentException();
    }
    if (columnFraction < 0) {
      columnFraction = 0;
    }
    if (columnFraction > 1) {
      columnFraction = 1;
    }

    // random generator seeded with current time
    if (randomGenerator == null) {
      randomGenerator = new MersenneTwister((int) System.currentTimeMillis());
    }

    int nslices = (int) Math.round(matrix.slices() * sliceFraction);
    int nrows = (int) Math.round(matrix.rows() * rowFraction);
    int ncols = (int) Math.round(matrix.columns() * columnFraction);
    int max = Math.max(nslices, Math.max(nrows, ncols));
    long[] selected = new long[max]; // sampler works on long's, not int's

    // sample slices
    int n = nslices;
    int N = matrix.slices();
    RandomSampler.sample(n, N, n, 0, selected, 0, randomGenerator);
    int[] selectedSlices = new int[n];
    for (int i = 0; i < n; i++) {
      selectedSlices[i] = (int) selected[i];
    }

    // sample rows
    n = nrows;
    N = matrix.rows();
    RandomSampler.sample(n, N, n, 0, selected, 0, randomGenerator);
    int[] selectedRows = new int[n];
    for (int i = 0; i < n; i++) {
      selectedRows[i] = (int) selected[i];
    }

    // sample columns
    n = ncols;
    N = matrix.columns();
    RandomSampler.sample(n, N, n, 0, selected, 0, randomGenerator);
    int[] selectedCols = new int[n];
    for (int i = 0; i < n; i++) {
      selectedCols[i] = (int) selected[i];
    }

    return matrix.viewSelection(selectedSlices, selectedRows, selectedCols);
  }

}
