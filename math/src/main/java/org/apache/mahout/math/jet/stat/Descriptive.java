/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.stat;

import org.apache.mahout.math.list.DoubleArrayList;
import org.apache.mahout.math.list.IntArrayList;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class Descriptive {

  /** Makes this class non instantiable, but still let's others inherit from it. */
  private Descriptive() {
  }

  /** Returns the auto-correlation of a data sequence. */
  public static double autoCorrelation(DoubleArrayList data, int lag, double mean, double variance) {
    int N = data.size();
    if (lag >= N) {
      throw new IllegalArgumentException("Lag is too large");
    }

    double[] elements = data.elements();
    double run = 0;
    for (int i = lag; i < N; ++i) {
      run += (elements[i] - mean) * (elements[i - lag] - mean);
    }

    return (run / (N - lag)) / variance;
  }

  /**
   * Checks if the given range is within the contained array's bounds.
   *
   * @throws IndexOutOfBoundsException if <tt>to!=from-1 || from&lt;0 || from&gt;to || to&gt;=size()</tt>.
   */
  protected static void checkRangeFromTo(int from, int to, int theSize) {
    if (to == from - 1) {
      return;
    }
    if (from < 0 || from > to || to >= theSize) {
      throw new IndexOutOfBoundsException("from: " + from + ", to: " + to + ", size=" + theSize);
    }
  }

  /** Returns the correlation of two data sequences. That is <tt>covariance(data1,data2)/(standardDev1*standardDev2)</tt>. */
  public static double correlation(DoubleArrayList data1, double standardDev1, DoubleArrayList data2,
                                   double standardDev2) {
    return covariance(data1, data2) / (standardDev1 * standardDev2);
  }

  /**
   * Returns the covariance of two data sequences, which is <tt>cov(x,y) = (1/(size()-1)) * Sum((x[i]-mean(x)) *
   * (y[i]-mean(y)))</tt>. See the <A HREF="http://www.cquest.utoronto.ca/geog/ggr270y/notes/not05efg.html"> math
   * definition</A>.
   */
  public static double covariance(DoubleArrayList data1, DoubleArrayList data2) {
    int size = data1.size();
    if (size != data2.size() || size == 0) {
      throw new IllegalArgumentException();
    }
    double[] elements1 = data1.elements();
    double[] elements2 = data2.elements();

    double sumx = elements1[0], sumy = elements2[0], Sxy = 0;
    for (int i = 1; i < size; ++i) {
      double x = elements1[i];
      double y = elements2[i];
      sumx += x;
      Sxy += (x - sumx / (i + 1)) * (y - sumy / i);
      sumy += y;
      // Exercise for the reader: Why does this give us the right answer?
    }
    return Sxy / (size - 1);
  }

  /*
  * Both covariance versions yield the same results but the one above is faster
  */

  private static double covariance2(DoubleArrayList data1, DoubleArrayList data2) {
    int size = data1.size();
    double mean1 = mean(data1);
    double mean2 = mean(data2);
    double covariance = 0.0D;
    for (int i = 0; i < size; i++) {
      double x = data1.get(i);
      double y = data2.get(i);

      covariance += (x - mean1) * (y - mean2);
    }

    return covariance / (double) (size - 1);
  }


  /** Durbin-Watson computation. */
  public static double durbinWatson(DoubleArrayList data) {
    int size = data.size();
    if (size < 2) {
      throw new IllegalArgumentException("data sequence must contain at least two values.");
    }

    double[] elements = data.elements();
    double run_sq = elements[0] * elements[0];
    double run = 0;
    for (int i = 1; i < size; ++i) {
      double x = elements[i] - elements[i - 1];
      run += x * x;
      run_sq += elements[i] * elements[i];
    }

    return run / run_sq;
  }

  /**
   * Computes the frequency (number of occurances, count) of each distinct value in the given sorted data. After this
   * call returns both <tt>distinctValues</tt> and <tt>frequencies</tt> have a new size (which is equal for both), which
   * is the number of distinct values in the sorted data. <p> Distinct values are filled into <tt>distinctValues</tt>,
   * starting at index 0. The frequency of each distinct value is filled into <tt>frequencies</tt>, starting at index 0.
   * As a result, the smallest distinct value (and its frequency) can be found at index 0, the second smallest distinct
   * value (and its frequency) at index 1, ..., the largest distinct value (and its frequency) at index
   * <tt>distinctValues.size()-1</tt>.
   *
   * <b>Example:</b> <br> <tt>elements = (5,6,6,7,8,8) --> distinctValues = (5,6,7,8), frequencies = (1,2,1,2)</tt>
   *
   * @param sortedData     the data; must be sorted ascending.
   * @param distinctValues a list to be filled with the distinct values; can have any size.
   * @param frequencies    a list to be filled with the frequencies; can have any size; set this parameter to
   *                       <tt>null</tt> to ignore it.
   */
  public static void frequencies(DoubleArrayList sortedData, DoubleArrayList distinctValues, IntArrayList frequencies) {
    distinctValues.clear();
    if (frequencies != null) {
      frequencies.clear();
    }

    double[] sortedElements = sortedData.elements();
    int size = sortedData.size();
    int i = 0;

    while (i < size) {
      double element = sortedElements[i];
      int cursor = i;

      // determine run length (number of equal elements)
      while (++i < size && sortedElements[i] == element) {
      }

      int runLength = i - cursor;
      distinctValues.add(element);
      if (frequencies != null) {
        frequencies.add(runLength);
      }
    }
  }

  /**
   * Returns the geometric mean of a data sequence. Note that for a geometric mean to be meaningful, the minimum of the
   * data sequence must not be less or equal to zero. <br> The geometric mean is given by <tt>pow( Product( data[i] ),
   * 1/size)</tt> which is equivalent to <tt>Math.exp( Sum( Log(data[i]) ) / size)</tt>.
   */
  public static double geometricMean(int size, double sumOfLogarithms) {
    return Math.exp(sumOfLogarithms / size);

    // this version would easily results in overflows
    //return Math.pow(product, 1/size);
  }

  /**
   * Returns the geometric mean of a data sequence. Note that for a geometric mean to be meaningful, the minimum of the
   * data sequence must not be less or equal to zero. <br> The geometric mean is given by <tt>pow( Product( data[i] ),
   * 1/data.size())</tt>. This method tries to avoid overflows at the expense of an equivalent but somewhat slow
   * definition: <tt>geo = Math.exp( Sum( Log(data[i]) ) / data.size())</tt>.
   */
  public static double geometricMean(DoubleArrayList data) {
    return geometricMean(data.size(), sumOfLogarithms(data, 0, data.size() - 1));
  }

  /**
   * Returns the harmonic mean of a data sequence.
   *
   * @param size            the number of elements in the data sequence.
   * @param sumOfInversions <tt>Sum( 1.0 / data[i])</tt>.
   */
  public static double harmonicMean(int size, double sumOfInversions) {
    return size / sumOfInversions;
  }

  /**
   * Incrementally maintains and updates minimum, maximum, sum and sum of squares of a data sequence.
   *
   * Assume we have already recorded some data sequence elements and know their minimum, maximum, sum and sum of
   * squares. Assume further, we are to record some more elements and to derive updated values of minimum, maximum, sum
   * and sum of squares. <p> This method computes those updated values without needing to know the already recorded
   * elements. This is interesting for interactive online monitoring and/or applications that cannot keep the entire
   * huge data sequence in memory. <p> <br>Definition of sumOfSquares: <tt>sumOfSquares(n) = Sum ( data[i] * data[i]
   * )</tt>.
   *
   * @param data  the additional elements to be incorporated into min, max, etc.
   * @param from  the index of the first element within <tt>data</tt> to consider.
   * @param to    the index of the last element within <tt>data</tt> to consider. The method incorporates elements
   *              <tt>data[from], ..., data[to]</tt>.
   * @param inOut the old values in the following format: <ul> <li><tt>inOut[0]</tt> is the old minimum.
   *              <li><tt>inOut[1]</tt> is the old maximum. <li><tt>inOut[2]</tt> is the old sum. <li><tt>inOut[3]</tt>
   *              is the old sum of squares. </ul> If no data sequence elements have so far been recorded set the values
   *              as follows <ul> <li><tt>inOut[0] = Double.POSITIVE_INFINITY</tt> as the old minimum. <li><tt>inOut[1]
   *              = Double.NEGATIVE_INFINITY</tt> as the old maximum. <li><tt>inOut[2] = 0.0</tt> as the old sum.
   *              <li><tt>inOut[3] = 0.0</tt> as the old sum of squares. </ul>
   * @return the updated values filled into the <tt>inOut</tt> array.
   */
  public static void incrementalUpdate(DoubleArrayList data, int from, int to, double[] inOut) {
    checkRangeFromTo(from, to, data.size());

    // read current values
    double min = inOut[0];
    double max = inOut[1];
    double sum = inOut[2];
    double sumSquares = inOut[3];

    double[] elements = data.elements();

    for (; from <= to; from++) {
      double element = elements[from];
      sum += element;
      sumSquares += element * element;
      if (element < min) {
        min = element;
      }
      if (element > max) {
        max = element;
      }

      /*
      double oldDeviation = element - mean;
      mean += oldDeviation / (N+1);
      sumSquaredDeviations += (element-mean)*oldDeviation; // cool, huh?
      */

      /*
      double oldMean = mean;
      mean += (element - mean)/(N+1);
      if (N > 0) {
        sumSquaredDeviations += (element-mean)*(element-oldMean); // cool, huh?
      }
      */

    }

    // store new values
    inOut[0] = min;
    inOut[1] = max;
    inOut[2] = sum;
    inOut[3] = sumSquares;

    // At this point of return the following postcondition holds:
    // data.size()-from elements have been consumed by this call.
  }

  /**
   * Incrementally maintains and updates various sums of powers of the form <tt>Sum(data[i]<sup>k</sup>)</tt>.
   *
   * Assume we have already recorded some data sequence elements <tt>data[i]</tt> and know the values of
   * <tt>Sum(data[i]<sup>from</sup>), Sum(data[i]<sup>from+1</sup>), ..., Sum(data[i]<sup>to</sup>)</tt>. Assume
   * further, we are to record some more elements and to derive updated values of these sums. <p> This method computes
   * those updated values without needing to know the already recorded elements. This is interesting for interactive
   * online monitoring and/or applications that cannot keep the entire huge data sequence in memory. For example, the
   * incremental computation of moments is based upon such sums of powers: <p> The moment of <tt>k</tt>-th order with
   * constant <tt>c</tt> of a data sequence, is given by <tt>Sum( (data[i]-c)<sup>k</sup> ) / data.size()</tt>. It can
   * incrementally be computed by using the equivalent formula <p> <tt>moment(k,c) = m(k,c) / data.size()</tt> where
   * <br><tt>m(k,c) = Sum( -1<sup>i</sup> * b(k,i) * c<sup>i</sup> * sumOfPowers(k-i))</tt> for <tt>i = 0 .. k</tt> and
   * <br><tt>b(k,i) = </tt>{@link org.apache.mahout.math.jet.math.Arithmetic#binomial(long,long) binomial(k,i)} and
   * <br><tt>sumOfPowers(k) = Sum( data[i]<sup>k</sup> )</tt>. <p>
   *
   * @param data        the additional elements to be incorporated into min, max, etc.
   * @param from        the index of the first element within <tt>data</tt> to consider.
   * @param to          the index of the last element within <tt>data</tt> to consider. The method incorporates elements
   *                    <tt>data[from], ..., data[to]</tt>.
   * @param sumOfPowers the old values of the sums in the following format: <ul> <li><tt>sumOfPowers[0]</tt> is the old
   *                    <tt>Sum(data[i]<sup>fromSumIndex</sup>)</tt>. <li><tt>sumOfPowers[1]</tt> is the old
   *                    <tt>Sum(data[i]<sup>fromSumIndex+1</sup>)</tt>. <li>... <li><tt>sumOfPowers[toSumIndex-fromSumIndex]</tt>
   *                    is the old <tt>Sum(data[i]<sup>toSumIndex</sup>)</tt>. </ul> If no data sequence elements have
   *                    so far been recorded set all old values of the sums to <tt>0.0</tt>.
   * @return the updated values filled into the <tt>sumOfPowers</tt> array.
   */
  public static void incrementalUpdateSumsOfPowers(DoubleArrayList data, int from, int to, int fromSumIndex,
                                                   int toSumIndex, double[] sumOfPowers) {
    int size = data.size();
    int lastIndex = toSumIndex - fromSumIndex;
    if (from > size || lastIndex + 1 > sumOfPowers.length) {
      throw new IllegalArgumentException();
    }

    // optimized for common parameters
    if (fromSumIndex == 1) { // handle quicker
      if (toSumIndex == 2) {
        double[] elements = data.elements();
        double sum = sumOfPowers[0];
        double sumSquares = sumOfPowers[1];
        for (int i = from - 1; ++i <= to;) {
          double element = elements[i];
          sum += element;
          sumSquares += element * element;
          //if (element < min) min = element;
          //else if (element > max) max = element;
        }
        sumOfPowers[0] += sum;
        sumOfPowers[1] += sumSquares;
        return;
      } else if (toSumIndex == 3) {
        double[] elements = data.elements();
        double sum = sumOfPowers[0];
        double sumSquares = sumOfPowers[1];
        double sum_xxx = sumOfPowers[2];
        for (int i = from - 1; ++i <= to;) {
          double element = elements[i];
          sum += element;
          sumSquares += element * element;
          sum_xxx += element * element * element;
          //if (element < min) min = element;
          //else if (element > max) max = element;
        }
        sumOfPowers[0] += sum;
        sumOfPowers[1] += sumSquares;
        sumOfPowers[2] += sum_xxx;
        return;
      } else if (toSumIndex == 4) { // handle quicker
        double[] elements = data.elements();
        double sum = sumOfPowers[0];
        double sumSquares = sumOfPowers[1];
        double sum_xxx = sumOfPowers[2];
        double sum_xxxx = sumOfPowers[3];
        for (int i = from - 1; ++i <= to;) {
          double element = elements[i];
          sum += element;
          sumSquares += element * element;
          sum_xxx += element * element * element;
          sum_xxxx += element * element * element * element;
          //if (element < min) min = element;
          //else if (element > max) max = element;
        }
        sumOfPowers[0] += sum;
        sumOfPowers[1] += sumSquares;
        sumOfPowers[2] += sum_xxx;
        sumOfPowers[3] += sum_xxxx;
        return;
      }
    }

    if (fromSumIndex == toSumIndex || (fromSumIndex >= -1 && toSumIndex <= 5)) { // handle quicker
      for (int i = fromSumIndex; i <= toSumIndex; i++) {
        sumOfPowers[i - fromSumIndex] += sumOfPowerDeviations(data, i, 0.0, from, to);
      }
      return;
    }


    // now the most general case:
    // optimized for maximum speed, but still not quite quick
    double[] elements = data.elements();

    for (int i = from - 1; ++i <= to;) {
      double element = elements[i];
      double pow = Math.pow(element, fromSumIndex);

      int j = 0;
      for (int m = lastIndex; --m >= 0;) {
        sumOfPowers[j++] += pow;
        pow *= element;
      }
      sumOfPowers[j] += pow;
    }

    // At this point of return the following postcondition holds:
    // data.size()-fromIndex elements have been consumed by this call.
  }

  /**
   * Incrementally maintains and updates sum and sum of squares of a <i>weighted</i> data sequence.
   *
   * Assume we have already recorded some data sequence elements and know their sum and sum of squares. Assume further,
   * we are to record some more elements and to derive updated values of sum and sum of squares. <p> This method
   * computes those updated values without needing to know the already recorded elements. This is interesting for
   * interactive online monitoring and/or applications that cannot keep the entire huge data sequence in memory. <p>
   * <br>Definition of sum: <tt>sum = Sum ( data[i] * weights[i] )</tt>. <br>Definition of sumOfSquares:
   * <tt>sumOfSquares = Sum ( data[i] * data[i] * weights[i])</tt>.
   *
   * @param data    the additional elements to be incorporated into min, max, etc.
   * @param weights the weight of each element within <tt>data</tt>.
   * @param from    the index of the first element within <tt>data</tt> (and <tt>weights</tt>) to consider.
   * @param to      the index of the last element within <tt>data</tt> (and <tt>weights</tt>) to consider. The method
   *                incorporates elements <tt>data[from], ..., data[to]</tt>.
   * @param inOut   the old values in the following format: <ul> <li><tt>inOut[0]</tt> is the old sum.
   *                <li><tt>inOut[1]</tt> is the old sum of squares. </ul> If no data sequence elements have so far been
   *                recorded set the values as follows <ul> <li><tt>inOut[0] = 0.0</tt> as the old sum. <li><tt>inOut[1]
   *                = 0.0</tt> as the old sum of squares. </ul>
   * @return the updated values filled into the <tt>inOut</tt> array.
   */
  public static void incrementalWeightedUpdate(DoubleArrayList data, DoubleArrayList weights, int from, int to,
                                               double[] inOut) {
    int dataSize = data.size();
    checkRangeFromTo(from, to, dataSize);
    if (dataSize != weights.size()) {
      throw new IllegalArgumentException(
          "from=" + from + ", to=" + to + ", data.size()=" + dataSize + ", weights.size()=" + weights.size());
    }

    // read current values
    double sum = inOut[0];
    double sumOfSquares = inOut[1];

    double[] elements = data.elements();
    double[] w = weights.elements();

    for (int i = from - 1; ++i <= to;) {
      double element = elements[i];
      double weight = w[i];
      double prod = element * weight;

      sum += prod;
      sumOfSquares += element * prod;
    }

    // store new values
    inOut[0] = sum;
    inOut[1] = sumOfSquares;

    // At this point of return the following postcondition holds:
    // data.size()-from elements have been consumed by this call.
  }

  /**
   * Returns the kurtosis (aka excess) of a data sequence.
   *
   * @param moment4           the fourth central moment, which is <tt>moment(data,4,mean)</tt>.
   * @param standardDeviation the standardDeviation.
   */
  public static double kurtosis(double moment4, double standardDeviation) {
    return -3 + moment4 / (standardDeviation * standardDeviation * standardDeviation * standardDeviation);
  }

  /**
   * Returns the kurtosis (aka excess) of a data sequence, which is <tt>-3 + moment(data,4,mean) /
   * standardDeviation<sup>4</sup></tt>.
   */
  public static double kurtosis(DoubleArrayList data, double mean, double standardDeviation) {
    return kurtosis(moment(data, 4, mean), standardDeviation);
  }

  /**
   * Returns the lag-1 autocorrelation of a dataset; Note that this method has semantics different from
   * <tt>autoCorrelation(..., 1)</tt>;
   */
  public static double lag1(DoubleArrayList data, double mean) {
    int size = data.size();
    double[] elements = data.elements();
    double q = 0;
    double v = (elements[0] - mean) * (elements[0] - mean);

    for (int i = 1; i < size; i++) {
      double delta0 = (elements[i - 1] - mean);
      double delta1 = (elements[i] - mean);
      q += (delta0 * delta1 - q) / (i + 1);
      v += (delta1 * delta1 - v) / (i + 1);
    }

    return q / v;
  }

  /** Returns the largest member of a data sequence. */
  public static double max(DoubleArrayList data) {
    int size = data.size();
    if (size == 0) {
      throw new IllegalArgumentException();
    }

    double[] elements = data.elements();
    double max = elements[size - 1];
    for (int i = size - 1; --i >= 0;) {
      if (elements[i] > max) {
        max = elements[i];
      }
    }

    return max;
  }

  /** Returns the arithmetic mean of a data sequence; That is <tt>Sum( data[i] ) / data.size()</tt>. */
  public static double mean(DoubleArrayList data) {
    return sum(data) / data.size();
  }

  /** Returns the mean deviation of a dataset. That is <tt>Sum (Math.abs(data[i]-mean)) / data.size())</tt>. */
  public static double meanDeviation(DoubleArrayList data, double mean) {
    double[] elements = data.elements();
    int size = data.size();
    double sum = 0;
    for (int i = size; --i >= 0;) {
      sum += Math.abs(elements[i] - mean);
    }
    return sum / size;
  }

  /**
   * Returns the median of a sorted data sequence.
   *
   * @param sortedData the data sequence; <b>must be sorted ascending</b>.
   */
  public static double median(DoubleArrayList sortedData) {
    return quantile(sortedData, 0.5);
    /*
    double[] sortedElements = sortedData.elements();
    int n = sortedData.size();
    int lhs = (n - 1) / 2 ;
    int rhs = n / 2 ;

    if (n == 0) return 0.0 ;

    double median;
    if (lhs == rhs) median = sortedElements[lhs] ;
    else median = (sortedElements[lhs] + sortedElements[rhs])/2.0 ;

    return median;
    */
  }

  /** Returns the smallest member of a data sequence. */
  public static double min(DoubleArrayList data) {
    int size = data.size();
    if (size == 0) {
      throw new IllegalArgumentException();
    }

    double[] elements = data.elements();
    double min = elements[size - 1];
    for (int i = size - 1; --i >= 0;) {
      if (elements[i] < min) {
        min = elements[i];
      }
    }

    return min;
  }

  /**
   * Returns the moment of <tt>k</tt>-th order with constant <tt>c</tt> of a data sequence, which is <tt>Sum(
   * (data[i]-c)<sup>k</sup> ) / data.size()</tt>.
   *
   * @param sumOfPowers <tt>sumOfPowers[m] == Sum( data[i]<sup>m</sup>) )</tt> for <tt>m = 0,1,..,k</tt> as returned by
   *                    method {@link #incrementalUpdateSumsOfPowers(DoubleArrayList,int,int,int,int,double[])}. In
   *                    particular there must hold <tt>sumOfPowers.length == k+1</tt>.
   * @param size        the number of elements of the data sequence.
   */
  public static double moment(int k, double c, int size, double[] sumOfPowers) {
    double sum = 0;
    int sign = 1;
    for (int i = 0; i <= k; i++) {
      double y;
      if (i == 0) {
        y = 1;
      } else if (i == 1) {
        y = c;
      } else if (i == 2) {
        y = c * c;
      } else if (i == 3) {
        y = c * c * c;
      } else {
        y = Math.pow(c, i);
      }
      //sum += sign *
      sum += sign * org.apache.mahout.math.jet.math.Arithmetic.binomial(k, i) * y * sumOfPowers[k - i];
      sign = -sign;
    }
    /*
    for (int i=0; i<=k; i++) {
      sum += sign * org.apache.mahout.math.jet.math.Arithmetic.binomial(k,i) * Math.pow(c, i) * sumOfPowers[k-i];
      sign = -sign;
    }
    */
    return sum / size;
  }

  /**
   * Returns the moment of <tt>k</tt>-th order with constant <tt>c</tt> of a data sequence, which is <tt>Sum(
   * (data[i]-c)<sup>k</sup> ) / data.size()</tt>.
   */
  public static double moment(DoubleArrayList data, int k, double c) {
    return sumOfPowerDeviations(data, k, c) / data.size();
  }

  /**
   * Returns the pooled mean of two data sequences. That is <tt>(size1 * mean1 + size2 * mean2) / (size1 + size2)</tt>.
   *
   * @param size1 the number of elements in data sequence 1.
   * @param mean1 the mean of data sequence 1.
   * @param size2 the number of elements in data sequence 2.
   * @param mean2 the mean of data sequence 2.
   */
  public static double pooledMean(int size1, double mean1, int size2, double mean2) {
    return (size1 * mean1 + size2 * mean2) / (size1 + size2);
  }

  /**
   * Returns the pooled variance of two data sequences. That is <tt>(size1 * variance1 + size2 * variance2) / (size1 +
   * size2)</tt>;
   *
   * @param size1     the number of elements in data sequence 1.
   * @param variance1 the variance of data sequence 1.
   * @param size2     the number of elements in data sequence 2.
   * @param variance2 the variance of data sequence 2.
   */
  public static double pooledVariance(int size1, double variance1, int size2, double variance2) {
    return (size1 * variance1 + size2 * variance2) / (size1 + size2);
  }

  /**
   * Returns the product, which is <tt>Prod( data[i] )</tt>. In other words: <tt>data[0]*data[1]*...*data[data.size()-1]</tt>.
   * This method uses the equivalent definition: <tt>prod = pow( exp( Sum( Log(x[i]) ) / size(), size())</tt>.
   */
  public static double product(int size, double sumOfLogarithms) {
    return Math.pow(Math.exp(sumOfLogarithms / size), size);
  }

  /**
   * Returns the product of a data sequence, which is <tt>Prod( data[i] )</tt>. In other words:
   * <tt>data[0]*data[1]*...*data[data.size()-1]</tt>. Note that you may easily get numeric overflows.
   */
  public static double product(DoubleArrayList data) {
    int size = data.size();
    double[] elements = data.elements();

    double product = 1;
    for (int i = size; --i >= 0;) {
      product *= elements[i];
    }

    return product;
  }

  /**
   * Returns the <tt>phi-</tt>quantile; that is, an element <tt>elem</tt> for which holds that <tt>phi</tt> percent of
   * data elements are less than <tt>elem</tt>. The quantile need not necessarily be contained in the data sequence, it
   * can be a linear interpolation.
   *
   * @param sortedData the data sequence; <b>must be sorted ascending</b>.
   * @param phi        the percentage; must satisfy <tt>0 &lt;= phi &lt;= 1</tt>.
   */
  public static double quantile(DoubleArrayList sortedData, double phi) {
    double[] sortedElements = sortedData.elements();
    int n = sortedData.size();

    double index = phi * (n - 1);
    int lhs = (int) index;
    double delta = index - lhs;

    if (n == 0) {
      return 0.0;
    }

    double result;
    if (lhs == n - 1) {
      result = sortedElements[lhs];
    } else {
      result = (1 - delta) * sortedElements[lhs] + delta * sortedElements[lhs + 1];
    }

    return result;
  }

  /**
   * Returns how many percent of the elements contained in the receiver are <tt>&lt;= element</tt>. Does linear
   * interpolation if the element is not contained but lies in between two contained elements.
   *
   * @param sortedList the list to be searched (must be sorted ascending).
   * @param element    the element to search for.
   * @return the percentage <tt>phi</tt> of elements <tt>&lt;= element</tt> (<tt>0.0 &lt;= phi &lt;= 1.0)</tt>.
   */
  public static double quantileInverse(DoubleArrayList sortedList, double element) {
    return rankInterpolated(sortedList, element) / sortedList.size();
  }

  /**
   * Returns the quantiles of the specified percentages. The quantiles need not necessarily be contained in the data
   * sequence, it can be a linear interpolation.
   *
   * @param sortedData  the data sequence; <b>must be sorted ascending</b>.
   * @param percentages the percentages for which quantiles are to be computed. Each percentage must be in the interval
   *                    <tt>[0.0,1.0]</tt>.
   * @return the quantiles.
   */
  public static DoubleArrayList quantiles(DoubleArrayList sortedData, DoubleArrayList percentages) {
    int s = percentages.size();
    DoubleArrayList quantiles = new DoubleArrayList(s);

    for (int i = 0; i < s; i++) {
      quantiles.add(quantile(sortedData, percentages.get(i)));
    }

    return quantiles;
  }

  /**
   * Returns the linearly interpolated number of elements in a list less or equal to a given element. The rank is the
   * number of elements <= element. Ranks are of the form <tt>{0, 1, 2,..., sortedList.size()}</tt>. If no element is <=
   * element, then the rank is zero. If the element lies in between two contained elements, then linear interpolation is
   * used and a non integer value is returned.
   *
   * @param sortedList the list to be searched (must be sorted ascending).
   * @param element    the element to search for.
   * @return the rank of the element.
   */
  public static double rankInterpolated(DoubleArrayList sortedList, double element) {
    int index = sortedList.binarySearch(element);
    if (index >= 0) { // element found
      // skip to the right over multiple occurances of element.
      int to = index + 1;
      int s = sortedList.size();
      while (to < s && sortedList.get(to) == element) {
        to++;
      }
      return to;
    }

    // element not found
    int insertionPoint = -index - 1;
    if (insertionPoint == 0 || insertionPoint == sortedList.size()) {
      return insertionPoint;
    }

    double from = sortedList.get(insertionPoint - 1);
    double to = sortedList.get(insertionPoint);
    double delta = (element - from) / (to - from); //linear interpolation
    return insertionPoint + delta;
  }

  /**
   * Returns the RMS (Root-Mean-Square) of a data sequence. That is <tt>Math.sqrt(Sum( data[i]*data[i] ) /
   * data.size())</tt>. The RMS of data sequence is the square-root of the mean of the squares of the elements in the
   * data sequence. It is a measure of the average "size" of the elements of a data sequence.
   *
   * @param sumOfSquares <tt>sumOfSquares(data) == Sum( data[i]*data[i] )</tt> of the data sequence.
   * @param size         the number of elements in the data sequence.
   */
  public static double rms(int size, double sumOfSquares) {
    return Math.sqrt(sumOfSquares / size);
  }

  /**
   * Returns the sample kurtosis (aka excess) of a data sequence.
   *
   * Ref: R.R. Sokal, F.J. Rohlf, Biometry: the principles and practice of statistics in biological research (W.H.
   * Freeman and Company, New York, 1998, 3rd edition) p. 114-115.
   *
   * @param size           the number of elements of the data sequence.
   * @param moment4        the fourth central moment, which is <tt>moment(data,4,mean)</tt>.
   * @param sampleVariance the <b>sample variance</b>.
   */
  public static double sampleKurtosis(int size, double moment4, double sampleVariance) {
    int n = size;
    double s2 = sampleVariance; // (y-ymean)^2/(n-1)
    double m4 = moment4 * n;    // (y-ymean)^4
    return m4 * n * (n + 1) / ((n - 1) * (n - 2) * (n - 3) * s2 * s2)
        - 3.0 * (n - 1) * (n - 1) / ((n - 2) * (n - 3));
  }

  /** Returns the sample kurtosis (aka excess) of a data sequence. */
  public static double sampleKurtosis(DoubleArrayList data, double mean, double sampleVariance) {
    return sampleKurtosis(data.size(), moment(data, 4, mean), sampleVariance);
  }

  /**
   * Return the standard error of the sample kurtosis.
   *
   * Ref: R.R. Sokal, F.J. Rohlf, Biometry: the principles and practice of statistics in biological research (W.H.
   * Freeman and Company, New York, 1998, 3rd edition) p. 138.
   *
   * @param size the number of elements of the data sequence.
   */
  public static double sampleKurtosisStandardError(int size) {
    int n = size;
    return Math.sqrt(24.0 * n * (n - 1) * (n - 1) / ((n - 3) * (n - 2) * (n + 3) * (n + 5)));
  }

  /**
   * Returns the sample skew of a data sequence.
   *
   * Ref: R.R. Sokal, F.J. Rohlf, Biometry: the principles and practice of statistics in biological research (W.H.
   * Freeman and Company, New York, 1998, 3rd edition) p. 114-115.
   *
   * @param size           the number of elements of the data sequence.
   * @param moment3        the third central moment, which is <tt>moment(data,3,mean)</tt>.
   * @param sampleVariance the <b>sample variance</b>.
   */
  public static double sampleSkew(int size, double moment3, double sampleVariance) {
    int n = size;
    double s = Math.sqrt(sampleVariance); // sqrt( (y-ymean)^2/(n-1) )
    double m3 = moment3 * n;    // (y-ymean)^3
    return n * m3 / ((n - 1) * (n - 2) * s * s * s);
  }

  /** Returns the sample skew of a data sequence. */
  public static double sampleSkew(DoubleArrayList data, double mean, double sampleVariance) {
    return sampleSkew(data.size(), moment(data, 3, mean), sampleVariance);
  }

  /**
   * Return the standard error of the sample skew.
   *
   * Ref: R.R. Sokal, F.J. Rohlf, Biometry: the principles and practice of statistics in biological research (W.H.
   * Freeman and Company, New York, 1998, 3rd edition) p. 138.
   *
   * @param size the number of elements of the data sequence.
   */
  public static double sampleSkewStandardError(int size) {
    int n = size;
    return Math.sqrt(6.0 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)));
  }

  /**
   * Returns the sample standard deviation.
   *
   * Ref: R.R. Sokal, F.J. Rohlf, Biometry: the principles and practice of statistics in biological research (W.H.
   * Freeman and Company, New York, 1998, 3rd edition) p. 53.
   *
   * @param size           the number of elements of the data sequence.
   * @param sampleVariance the <b>sample variance</b>.
   */
  public static double sampleStandardDeviation(int size, double sampleVariance) {
    int n = size;

    // The standard deviation calculated as the sqrt of the variance underestimates
    // the unbiased standard deviation.
    double s = Math.sqrt(sampleVariance);
    // It needs to be multiplied by this correction factor.
    double Cn;
    if (n > 30) {
      Cn = 1 + 1.0 / (4 * (n - 1)); // Cn = 1+1/(4*(n-1));
    } else {
      Cn = Math.sqrt((n - 1) * 0.5) * Gamma.gamma((n - 1) * 0.5) / Gamma.gamma(n * 0.5);
    }
    return Cn * s;
  }

  /**
   * Returns the sample variance of a data sequence. That is <tt>(sumOfSquares - mean*sum) / (size - 1)</tt> with
   * <tt>mean = sum/size</tt>.
   *
   * @param size         the number of elements of the data sequence.
   * @param sum          <tt>== Sum( data[i] )</tt>.
   * @param sumOfSquares <tt>== Sum( data[i]*data[i] )</tt>.
   */
  public static double sampleVariance(int size, double sum, double sumOfSquares) {
    double mean = sum / size;
    return (sumOfSquares - mean * sum) / (size - 1);
  }

  /** Returns the sample variance of a data sequence. That is <tt>Sum ( (data[i]-mean)^2 ) / (data.size()-1)</tt>. */
  public static double sampleVariance(DoubleArrayList data, double mean) {
    double[] elements = data.elements();
    int size = data.size();
    double sum = 0;
    // find the sum of the squares
    for (int i = size; --i >= 0;) {
      double delta = elements[i] - mean;
      sum += delta * delta;
    }

    return sum / (size - 1);
  }

  /**
   * Returns the sample weighted variance of a data sequence. That is <tt>(sumOfSquaredProducts  -  sumOfProducts *
   * sumOfProducts / sumOfWeights) / (sumOfWeights - 1)</tt>.
   *
   * @param sumOfWeights         <tt>== Sum( weights[i] )</tt>.
   * @param sumOfProducts        <tt>== Sum( data[i] * weights[i] )</tt>.
   * @param sumOfSquaredProducts <tt>== Sum( data[i] * data[i] * weights[i] )</tt>.
   */
  public static double sampleWeightedVariance(double sumOfWeights, double sumOfProducts, double sumOfSquaredProducts) {
    return (sumOfSquaredProducts - sumOfProducts * sumOfProducts / sumOfWeights) / (sumOfWeights - 1);
  }

  /**
   * Returns the skew of a data sequence.
   *
   * @param moment3           the third central moment, which is <tt>moment(data,3,mean)</tt>.
   * @param standardDeviation the standardDeviation.
   */
  public static double skew(double moment3, double standardDeviation) {
    return moment3 / (standardDeviation * standardDeviation * standardDeviation);
  }

  /** Returns the skew of a data sequence, which is <tt>moment(data,3,mean) / standardDeviation<sup>3</sup></tt>. */
  public static double skew(DoubleArrayList data, double mean, double standardDeviation) {
    return skew(moment(data, 3, mean), standardDeviation);
  }

  /**
   * Splits (partitions) a list into sublists such that each sublist contains the elements with a given range.
   * <tt>splitters=(a,b,c,...,y,z)</tt> defines the ranges <tt>[-inf,a), [a,b), [b,c), ..., [y,z), [z,inf]</tt>.
   * <p><b>Examples:</b><br> <ul> <tt>data = (1,2,3,4,5,8,8,8,10,11)</tt>. <br><tt>splitters=(2,8)</tt> yields 3 bins:
   * <tt>(1), (2,3,4,5) (8,8,8,10,11)</tt>. <br><tt>splitters=()</tt> yields 1 bin: <tt>(1,2,3,4,5,8,8,8,10,11)</tt>.
   * <br><tt>splitters=(-5)</tt> yields 2 bins: <tt>(), (1,2,3,4,5,8,8,8,10,11)</tt>. <br><tt>splitters=(100)</tt>
   * yields 2 bins: <tt>(1,2,3,4,5,8,8,8,10,11), ()</tt>. </ul>
   *
   * @param sortedList the list to be partitioned (must be sorted ascending).
   * @param splitters  the points at which the list shall be partitioned (must be sorted ascending).
   * @return the sublists (an array with <tt>length == splitters.size() + 1</tt>. Each sublist is returned sorted
   *         ascending.
   */
  public static DoubleArrayList[] split(DoubleArrayList sortedList, DoubleArrayList splitters) {
    // assertion: data is sorted ascending.
    // assertion: splitValues is sorted ascending.
    int noOfBins = splitters.size() + 1;

    DoubleArrayList[] bins = new DoubleArrayList[noOfBins];
    for (int i = noOfBins; --i >= 0;) {
      bins[i] = new DoubleArrayList();
    }

    int listSize = sortedList.size();
    int nextStart = 0;
    int i = 0;
    while (nextStart < listSize && i < noOfBins - 1) {
      double splitValue = splitters.get(i);
      int index = sortedList.binarySearch(splitValue);
      if (index < 0) { // splitValue not found
        int insertionPosition = -index - 1;
        bins[i].addAllOfFromTo(sortedList, nextStart, insertionPosition - 1);
        nextStart = insertionPosition;
      } else { // splitValue found
        // For multiple identical elements ("runs"), binarySearch does not define which of all valid indexes is returned.
        // Thus, skip over to the first element of a run.
        do {
          index--;
        } while (index >= 0 && sortedList.get(index) == splitValue);

        bins[i].addAllOfFromTo(sortedList, nextStart, index);
        nextStart = index + 1;
      }
      i++;
    }

    // now fill the remainder
    bins[noOfBins - 1].addAllOfFromTo(sortedList, nextStart, sortedList.size() - 1);

    return bins;
  }

  /** Returns the standard deviation from a variance. */
  public static double standardDeviation(double variance) {
    return Math.sqrt(variance);
  }

  /**
   * Returns the standard error of a data sequence. That is <tt>Math.sqrt(variance/size)</tt>.
   *
   * @param size     the number of elements in the data sequence.
   * @param variance the variance of the data sequence.
   */
  public static double standardError(int size, double variance) {
    return Math.sqrt(variance / size);
  }

  /**
   * Modifies a data sequence to be standardized. Changes each element <tt>data[i]</tt> as follows: <tt>data[i] =
   * (data[i]-mean)/standardDeviation</tt>.
   */
  public static void standardize(DoubleArrayList data, double mean, double standardDeviation) {
    double[] elements = data.elements();
    for (int i = data.size(); --i >= 0;) {
      elements[i] = (elements[i] - mean) / standardDeviation;
    }
  }

  /** Returns the sum of a data sequence. That is <tt>Sum( data[i] )</tt>. */
  public static double sum(DoubleArrayList data) {
    return sumOfPowerDeviations(data, 1, 0.0);
  }

  /**
   * Returns the sum of inversions of a data sequence, which is <tt>Sum( 1.0 / data[i])</tt>.
   *
   * @param data the data sequence.
   * @param from the index of the first data element (inclusive).
   * @param to   the index of the last data element (inclusive).
   */
  public static double sumOfInversions(DoubleArrayList data, int from, int to) {
    return sumOfPowerDeviations(data, -1, 0.0, from, to);
  }

  /**
   * Returns the sum of logarithms of a data sequence, which is <tt>Sum( Log(data[i])</tt>.
   *
   * @param data the data sequence.
   * @param from the index of the first data element (inclusive).
   * @param to   the index of the last data element (inclusive).
   */
  public static double sumOfLogarithms(DoubleArrayList data, int from, int to) {
    double[] elements = data.elements();
    double logsum = 0;
    for (int i = from - 1; ++i <= to;) {
      logsum += Math.log(elements[i]);
    }
    return logsum;
  }

  /**
   * Returns <tt>Sum( (data[i]-c)<sup>k</sup> )</tt>; optimized for common parameters like <tt>c == 0.0</tt> and/or
   * <tt>k == -2 .. 4</tt>.
   */
  public static double sumOfPowerDeviations(DoubleArrayList data, int k, double c) {
    return sumOfPowerDeviations(data, k, c, 0, data.size() - 1);
  }

  /**
   * Returns <tt>Sum( (data[i]-c)<sup>k</sup> )</tt> for all <tt>i = from .. to</tt>; optimized for common parameters
   * like <tt>c == 0.0</tt> and/or <tt>k == -2 .. 5</tt>.
   */
  public static double sumOfPowerDeviations(DoubleArrayList data, int k, double c, int from, int to) {
    double[] elements = data.elements();
    double sum = 0;
    double v;
    int i;
    switch (k) { // optimized for speed
      case -2:
        if (c == 0.0) {
          for (i = from - 1; ++i <= to;) {
            v = elements[i];
            sum += 1 / (v * v);
          }
        } else {
          for (i = from - 1; ++i <= to;) {
            v = elements[i] - c;
            sum += 1 / (v * v);
          }
        }
        break;
      case -1:
        if (c == 0.0) {
          for (i = from - 1; ++i <= to;) {
            sum += 1 / (elements[i]);
          }
        } else {
          for (i = from - 1; ++i <= to;) {
            sum += 1 / (elements[i] - c);
          }
        }
        break;
      case 0:
        sum += to - from + 1;
        break;
      case 1:
        if (c == 0.0) {
          for (i = from - 1; ++i <= to;) {
            sum += elements[i];
          }
        } else {
          for (i = from - 1; ++i <= to;) {
            sum += elements[i] - c;
          }
        }
        break;
      case 2:
        if (c == 0.0) {
          for (i = from - 1; ++i <= to;) {
            v = elements[i];
            sum += v * v;
          }
        } else {
          for (i = from - 1; ++i <= to;) {
            v = elements[i] - c;
            sum += v * v;
          }
        }
        break;
      case 3:
        if (c == 0.0) {
          for (i = from - 1; ++i <= to;) {
            v = elements[i];
            sum += v * v * v;
          }
        } else {
          for (i = from - 1; ++i <= to;) {
            v = elements[i] - c;
            sum += v * v * v;
          }
        }
        break;
      case 4:
        if (c == 0.0) {
          for (i = from - 1; ++i <= to;) {
            v = elements[i];
            sum += v * v * v * v;
          }
        } else {
          for (i = from - 1; ++i <= to;) {
            v = elements[i] - c;
            sum += v * v * v * v;
          }
        }
        break;
      case 5:
        if (c == 0.0) {
          for (i = from - 1; ++i <= to;) {
            v = elements[i];
            sum += v * v * v * v * v;
          }
        } else {
          for (i = from - 1; ++i <= to;) {
            v = elements[i] - c;
            sum += v * v * v * v * v;
          }
        }
        break;
      default:
        for (i = from - 1; ++i <= to;) {
          sum += Math.pow(elements[i] - c, k);
        }
        break;
    }
    return sum;
  }

  /** Returns the sum of powers of a data sequence, which is <tt>Sum ( data[i]<sup>k</sup> )</tt>. */
  public static double sumOfPowers(DoubleArrayList data, int k) {
    return sumOfPowerDeviations(data, k, 0);
  }

  /**
   * Returns the sum of squared mean deviation of of a data sequence. That is <tt>variance * (size-1) == Sum( (data[i] -
   * mean)^2 )</tt>.
   *
   * @param size     the number of elements of the data sequence.
   * @param variance the variance of the data sequence.
   */
  public static double sumOfSquaredDeviations(int size, double variance) {
    return variance * (size - 1);
  }

  /** Returns the sum of squares of a data sequence. That is <tt>Sum ( data[i]*data[i] )</tt>. */
  public static double sumOfSquares(DoubleArrayList data) {
    return sumOfPowerDeviations(data, 2, 0.0);
  }

  /**
   * Returns the trimmed mean of a sorted data sequence.
   *
   * @param sortedData the data sequence; <b>must be sorted ascending</b>.
   * @param mean       the mean of the (full) sorted data sequence.
   */
  public static double trimmedMean(DoubleArrayList sortedData, double mean, int left, int right) {
    int N = sortedData.size();
    if (N == 0) {
      throw new IllegalArgumentException("Empty data.");
    }
    if (left + right >= N) {
      throw new IllegalArgumentException("Not enough data.");
    }

    double[] sortedElements = sortedData.elements();
    int N0 = N;
    for (int i = 0; i < left; ++i) {
      mean += (mean - sortedElements[i]) / (--N);
    }
    for (int i = 0; i < right; ++i) {
      mean += (mean - sortedElements[N0 - 1 - i]) / (--N);
    }
    return mean;
  }

  /** Returns the variance from a standard deviation. */
  public static double variance(double standardDeviation) {
    return standardDeviation * standardDeviation;
  }

  /**
   * Returns the variance of a data sequence. That is <tt>(sumOfSquares - mean*sum) / size</tt> with <tt>mean =
   * sum/size</tt>.
   *
   * @param size         the number of elements of the data sequence.
   * @param sum          <tt>== Sum( data[i] )</tt>.
   * @param sumOfSquares <tt>== Sum( data[i]*data[i] )</tt>.
   */
  public static double variance(int size, double sum, double sumOfSquares) {
    double mean = sum / size;
    return (sumOfSquares - mean * sum) / size;
  }

  /** Returns the weighted mean of a data sequence. That is <tt> Sum (data[i] * weights[i]) / Sum ( weights[i] )</tt>. */
  public static double weightedMean(DoubleArrayList data, DoubleArrayList weights) {
    int size = data.size();
    if (size != weights.size() || size == 0) {
      throw new IllegalArgumentException();
    }

    double[] elements = data.elements();
    double[] theWeights = weights.elements();
    double sum = 0.0;
    double weightsSum = 0.0;
    for (int i = size; --i >= 0;) {
      double w = theWeights[i];
      sum += elements[i] * w;
      weightsSum += w;
    }

    return sum / weightsSum;
  }

  /**
   * Returns the weighted RMS (Root-Mean-Square) of a data sequence. That is <tt>Sum( data[i] * data[i] * weights[i]) /
   * Sum( data[i] * weights[i] )</tt>, or in other words <tt>sumOfProducts / sumOfSquaredProducts</tt>.
   *
   * @param sumOfProducts        <tt>== Sum( data[i] * weights[i] )</tt>.
   * @param sumOfSquaredProducts <tt>== Sum( data[i] * data[i] * weights[i] )</tt>.
   */
  public static double weightedRMS(double sumOfProducts, double sumOfSquaredProducts) {
    return sumOfProducts / sumOfSquaredProducts;
  }

  /**
   * Returns the winsorized mean of a sorted data sequence.
   *
   * @param sortedData the data sequence; <b>must be sorted ascending</b>.
   * @param mean       the mean of the (full) sorted data sequence.
   */
  public static double winsorizedMean(DoubleArrayList sortedData, double mean, int left, int right) {
    int N = sortedData.size();
    if (N == 0) {
      throw new IllegalArgumentException("Empty data.");
    }
    if (left + right >= N) {
      throw new IllegalArgumentException("Not enough data.");
    }

    double[] sortedElements = sortedData.elements();

    double leftElement = sortedElements[left];
    for (int i = 0; i < left; ++i) {
      mean += (leftElement - sortedElements[i]) / N;
    }

    double rightElement = sortedElements[N - 1 - right];
    for (int i = 0; i < right; ++i) {
      mean += (rightElement - sortedElements[N - 1 - i]) / N;
    }

    return mean;
  }
}
