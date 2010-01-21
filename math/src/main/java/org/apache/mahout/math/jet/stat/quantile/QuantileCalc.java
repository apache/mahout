/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.stat.quantile;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Computes b and k vor various parameters. */
class QuantileCalc {

  private static final Logger log = LoggerFactory.getLogger(QuantileCalc.class);

  private QuantileCalc() {
  }

  /**
   * Efficiently computes the binomial coefficient, often also referred to as "n over k" or "n choose k". The binomial
   * coefficient is defined as n!/((n-k)!*k!). Tries to avoid numeric overflows.
   *
   * @return the binomial coefficient.
   */
  public static double binomial(long n, long k) {
    if (k == 0 || k == n) {
      return 1.0;
    }

    // since binomial(n,k)==binomial(n,n-k), we can enforce the faster variant,
    // which is also the variant minimizing number overflows.
    if (k > n / 2.0) {
      k = n - k;
    }

    double binomial = 1.0;
    long N = n - k + 1;
    for (long i = k; i > 0;) {
      binomial *= ((double) N++) / (double) (i--);
    }
    return binomial;
  }

  /**
   * Returns the smallest <code>long &gt;= value</code>. <dt>Examples: <code>1.0 -> 1, 1.2 -> 2, 1.9 -> 2</code>. This
   * method is safer than using (long) Math.ceil(value), because of possible rounding error.
   */
  public static long ceiling(double value) {
    return Math.round(Math.ceil(value));
  }

  /**
   * Computes the number of buffers and number of values per buffer such that quantiles can be determined with an
   * approximation error no more than epsilon with a certain probability.
   *
   * Assumes that quantiles are to be computed over N values. The required sampling rate is computed and stored in the
   * first element of the provided <tt>returnSamplingRate</tt> array, which, therefore must be at least of length 1.
   *
   * @param N                  the number of values over which quantiles shall be computed (e.g <tt>10^6</tt>).
   * @param epsilon            the approximation error which is guaranteed not to be exceeded (e.g. <tt>0.001</tt>)
   *                           (<tt>0 &lt;= epsilon &lt;= 1</tt>). To get exact result, set <tt>epsilon=0.0</tt>;
   * @param delta              the probability that the approximation error is more than than epsilon (e.g.
   *                           <tt>0.0001</tt>) (<tt>0 &lt;= delta &lt;= 1</tt>). To avoid probabilistic answers, set
   *                           <tt>delta=0.0</tt>.
   * @param quantiles          the number of quantiles to be computed (e.g. <tt>100</tt>) (<tt>quantiles &gt;= 1</tt>).
   *                           If unknown in advance, set this number large, e.g. <tt>quantiles &gt;= 10000</tt>.
   * @param returnSamplingRate a <tt>double[1]</tt> where the sampling rate is to be filled in.
   * @return <tt>long[2]</tt> - <tt>long[0]</tt>=the number of buffers, <tt>long[1]</tt>=the number of elements per
   *         buffer, <tt>returnSamplingRate[0]</tt>=the required sampling rate.
   */
  public static long[] known_N_compute_B_and_K(long N, double epsilon, double delta, int quantiles,
                                               double[] returnSamplingRate) {
    if (delta > 0.0) {
      return known_N_compute_B_and_K_slow(N, epsilon, delta, quantiles, returnSamplingRate);
    }
    returnSamplingRate[0] = 1.0;
    return known_N_compute_B_and_K_quick(N, epsilon);
  }

  /**
   * Computes the number of buffers and number of values per buffer such that quantiles can be determined with a
   * <b>guaranteed</b> approximation error no more than epsilon. Assumes that quantiles are to be computed over N
   * values.
   *
   * @param N       the anticipated number of values over which quantiles shall be determined.
   * @param epsilon the approximation error which is guaranteed not to be exceeded (e.g. <tt>0.001</tt>) (<tt>0 &lt;=
   *                epsilon &lt;= 1</tt>). To get exact result, set <tt>epsilon=0.0</tt>;
   * @return <tt>long[2]</tt> - <tt>long[0]</tt>=the number of buffers, <tt>long[1]</tt>=the number of elements per
   *         buffer.
   */
  protected static long[] known_N_compute_B_and_K_quick(long N, double epsilon) {
    if (epsilon <= 0.0) {
      // no way around exact quantile search
      long[] result = new long[2];
      result[0] = 1;
      result[1] = N;
      return result;
    }

    int maxBuffers = 50;
    int maxHeight = 50;
    double N_double = (double) N;
    double c = N_double * epsilon * 2.0;
    int[] heightMaximums = new int[maxBuffers - 1];

    // for each b, determine maximum height, i.e. the height for which x<=0 and x is a maximum
    // with x = binomial(b+h-2, h-1) - binomial(b+h-3, h-3) + binomial(b+h-3, h-2) - N * epsilon * 2.0
    for (int b = 2; b <= maxBuffers; b++) {
      int h = 3;

      while (h <= maxHeight && // skip heights until x<=0
          (h - 2) * ((double) Math.round(binomial(b + h - 2, h - 1))) -
              ((double) Math.round(binomial(b + h - 3, h - 3))) +
              ((double) Math.round(binomial(b + h - 3, h - 2))) - c
              > 0.0
          ) {
        h++;
      }
      //from now on x is monotonically growing...
      while (h <= maxHeight && // skip heights until x>0
          (h - 2) * ((double) Math.round(binomial(b + h - 2, h - 1))) -
              ((double) Math.round(binomial(b + h - 3, h - 3))) +
              ((double) Math.round(binomial(b + h - 3, h - 2))) - c
              <= 0.0
          ) {
        h++;
      }
      h--; //go back to last height

      // was x>0 or did we loop without finding anything?
      int hMax;
      if (h >= maxHeight &&
          (h - 2) * ((double) Math.round(binomial(b + h - 2, h - 1))) -
              ((double) Math.round(binomial(b + h - 3, h - 3))) +
              ((double) Math.round(binomial(b + h - 3, h - 2))) - c
              > 0.0) {
        hMax = Integer.MIN_VALUE;
      } else {
        hMax = h;
      }

      heightMaximums[b - 2] = hMax; //safe some space
    } //end for


    // for each b, determine the smallest k satisfying the constraints, i.e.
    // for each b, determine kMin, with kMin = N/binomial(b+hMax-2,hMax-1)
    long[] kMinimums = new long[maxBuffers - 1];
    for (int b = 2; b <= maxBuffers; b++) {
      int h = heightMaximums[b - 2];
      long kMin = Long.MAX_VALUE;
      if (h > Integer.MIN_VALUE) {
        double value = ((double) Math.round(binomial(b + h - 2, h - 1)));
        long tmpK = ceiling(N_double / value);
        if (tmpK <= kMin) {
          kMin = tmpK;
        }
      }
      kMinimums[b - 2] = kMin;
    }

    // from all b's, determine b that minimizes b*kMin
    long multMin = Long.MAX_VALUE;
    int minB = -1;
    for (int b = 2; b <= maxBuffers; b++) {
      if (kMinimums[b - 2] < Long.MAX_VALUE) {
        long mult = ((long) b) * kMinimums[b - 2];
        if (mult < multMin) {
          multMin = mult;
          minB = b;
        }
      }
    }

    long b, k;
    if (minB != -1) { // epsilon large enough?
      b = minB;
      k = kMinimums[minB - 2];
    } else {     // epsilon is very small or zero.
      b = 1; // the only possible solution without violating the
      k = N; // approximation guarantees is exact quantile search.
    }

    long[] result = new long[2];
    result[0] = b;
    result[1] = k;
    return result;
  }

  /**
   * Computes the number of buffers and number of values per buffer such that quantiles can be determined with an
   * approximation error no more than epsilon with a certain probability. Assumes that quantiles are to be computed over
   * N values. The required sampling rate is computed and stored in the first element of the provided
   * <tt>returnSamplingRate</tt> array, which, therefore must be at least of length 1.
   *
   * @param N                  the anticipated number of values over which quantiles shall be computed (e.g 10^6).
   * @param epsilon            the approximation error which is guaranteed not to be exceeded (e.g. <tt>0.001</tt>)
   *                           (<tt>0 &lt;= epsilon &lt;= 1</tt>). To get exact result, set <tt>epsilon=0.0</tt>;
   * @param delta              the probability that the approximation error is more than than epsilon (e.g.
   *                           <tt>0.0001</tt>) (<tt>0 &lt;= delta &lt;= 1</tt>). To avoid probabilistic answers, set
   *                           <tt>delta=0.0</tt>.
   * @param quantiles          the number of quantiles to be computed (e.g. <tt>100</tt>) (<tt>quantiles &gt;= 1</tt>).
   *                           If unknown in advance, set this number large, e.g. <tt>quantiles &gt;= 10000</tt>.
   * @param returnSamplingRate a <tt>double[1]</tt> where the sampling rate is to be filled in.
   * @return <tt>long[2]</tt> - <tt>long[0]</tt>=the number of buffers, <tt>long[1]</tt>=the number of elements per
   *         buffer, <tt>returnSamplingRate[0]</tt>=the required sampling rate.
   */
  protected static long[] known_N_compute_B_and_K_slow(long N, double epsilon, double delta, int quantiles,
                                                       double[] returnSamplingRate) {
    // delta can be set to zero, i.e., all quantiles should be approximate with probability 1
    if (epsilon <= 0.0) {
      // no way around exact quantile search
      long[] result = new long[2];
      result[0] = 1;
      result[1] = N;
      returnSamplingRate[0] = 1.0;
      return result;
    }


    int maxBuffers = 50;
    int maxHeight = 50;
    double N_double = N;

    // One possibility is to use one buffer of size N
    //
    long ret_b = 1;
    long ret_k = N;
    double sampling_rate = 1.0;
    long memory = N;


    // Otherwise, there are at least two buffers (b >= 2)
    // and the height of the tree is at least three (h >= 3)
    //
    // We restrict the search for b and h to MAX_BINOM, a large enough value for
    // practical values of    epsilon >= 0.001   and    delta >= 0.00001
    //
    double logarithm = Math.log(2.0 * quantiles / delta);
    double c = 2.0 * epsilon * N_double;
    for (long b = 2; b < maxBuffers; b++) {
      for (long h = 3; h < maxHeight; h++) {
        double binomial = binomial(b + h - 2, h - 1);
        long tmp = ceiling(N_double / binomial);
        if ((b * tmp < memory) &&
            ((h - 2) * binomial - binomial(b + h - 3, h - 3) + binomial(b + h - 3, h - 2)
                <= c)) {
          ret_k = tmp;
          ret_b = b;
          memory = ret_k * b;
          sampling_rate = 1.0;
        }
        if (delta > 0.0) {
          double t = (h - 2) * binomial(b + h - 2, h - 1) - binomial(b + h - 3, h - 3) + binomial(b + h - 3, h - 2);
          double u = logarithm / epsilon;
          double v = binomial(b + h - 2, h - 1);
          double w = logarithm / (2.0 * epsilon * epsilon);

          // From our SIGMOD 98 paper, we have two equantions to satisfy:
          // t  <= u * alpha/(1-alpha)^2
          // kv >= w/(1-alpha)^2
          //
          // Denoting 1/(1-alpha)    by x,
          // we see that the first inequality is equivalent to
          // t/u <= x^2 - x
          // which is satisfied by x >= 0.5 + 0.5 * sqrt (1 + 4t/u)
          // Plugging in this value into second equation yields
          // k >= wx^2/v

          double x = 0.5 + 0.5 * Math.sqrt(1.0 + 4.0 * t / u);
          long k = ceiling(w * x * x / v);
          if (b * k < memory) {
            ret_k = k;
            ret_b = b;
            memory = b * k;
            sampling_rate = N_double * 2.0 * epsilon * epsilon / logarithm;
          }
        }
      }
    }

    long[] result = new long[2];
    result[0] = ret_b;
    result[1] = ret_k;
    returnSamplingRate[0] = sampling_rate;
    return result;
  }

  public static void main(String[] args) {
    test_B_and_K_Calculation(args);
  }

  /** Computes b and k for different parameters. */
  public static void test_B_and_K_Calculation(String[] args) {
    boolean known_N;
    if (args == null) {
      known_N = false;
    } else {
      known_N = Boolean.valueOf(args[0]);
    }

    int[] quantiles = {1, 1000};

    long[] sizes = {100000, 1000000, 10000000, 1000000000};

    double[] deltas = {0.0, 0.001, 0.0001, 0.00001};

    double[] epsilons = {0.0, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0000001};


    if (!known_N) {
      sizes = new long[]{0};
    }
    log.info("\n\n");
    if (known_N) {
      log.info("Computing b's and k's for KNOWN N");
    } else {
      log.info("Computing b's and k's for UNKNOWN N");
    }
    log.info("mem [elements/1024]");
    log.info("***********************************");

    for (int p : quantiles) {
      log.info("------------------------------");
      log.info("computing for p = {}", p);
      for (long N : sizes) {
        log.info("   ------------------------------");
        log.info("   computing for N = {}", N);
        for (double delta : deltas) {
          log.info("      ------------------------------");
          log.info("      computing for delta = {}", delta);
          for (double epsilon : epsilons) {
            double[] returnSamplingRate = new double[1];
            long[] result;
            if (known_N) {
              result = known_N_compute_B_and_K(N, epsilon, delta, p, returnSamplingRate);
            } else {
              result = unknown_N_compute_B_and_K(epsilon, delta, p);
            }

            long b = result[0];
            long k = result[1];
            log.info("         (e,d,N,p)=({},{},{},{}) --> ", new Object[] {epsilon, delta, N, p});
            log.info("(b,k,mem");
            if (known_N) {
              log.info(",sampling");
            }
            log.info(")=({},{},{}", new Object[] {b, k, (b * k / 1024)});
            if (known_N) {
              log.info(",{}", returnSamplingRate[0]);
            }
            log.info(")");
          }
        }
      }
    }

  }

  /**
   * Computes the number of buffers and number of values per buffer such that quantiles can be determined with an
   * approximation error no more than epsilon with a certain probability.
   *
   * @param epsilon   the approximation error which is guaranteed not to be exceeded (e.g. <tt>0.001</tt>) (<tt>0 &lt;=
   *                  epsilon &lt;= 1</tt>). To get exact results, set <tt>epsilon=0.0</tt>;
   * @param delta     the probability that the approximation error is more than than epsilon (e.g. <tt>0.0001</tt>)
   *                  (<tt>0 &lt;= delta &lt;= 1</tt>). To get exact results, set <tt>delta=0.0</tt>.
   * @param quantiles the number of quantiles to be computed (e.g. <tt>100</tt>) (<tt>quantiles &gt;= 1</tt>). If
   *                  unknown in advance, set this number large, e.g. <tt>quantiles &gt;= 10000</tt>.
   * @return <tt>long[3]</tt> - <tt>long[0]</tt>=the number of buffers, <tt>long[1]</tt>=the number of elements per
   *         buffer, <tt>long[2]</tt>=the tree height where sampling shall start.
   */
  public static long[] unknown_N_compute_B_and_K(double epsilon, double delta, int quantiles) {
    // delta can be set to zero, i.e., all quantiles should be approximate with probability 1
    if (epsilon <= 0.0 || delta <= 0.0) {
      // no way around exact quantile search
      long[] result = new long[3];
      result[0] = 1;
      result[1] = Long.MAX_VALUE;
      result[2] = Long.MAX_VALUE;
      return result;
    }

    int max_b = 50;
    int max_h = 50;
    int max_H = 50;
    int max_Iterations = 2;

    long best_b = Long.MAX_VALUE;
    long best_k = Long.MAX_VALUE;
    long best_h = Long.MAX_VALUE;
    long best_memory = Long.MAX_VALUE;

    double pow = Math.pow(2.0, max_H);
    double logDelta = Math.log(2.0 / (delta / quantiles)) / (2.0 * epsilon * epsilon);
    //double logDelta =  Math.log(2.0/(quantiles*delta)) / (2.0*epsilon*epsilon);

    while (best_b == Long.MAX_VALUE && max_Iterations-- > 0) { //until we find a solution
      // identify that combination of b and h that minimizes b*k.
      // exhaustive search.
      for (int b = 2; b <= max_b; b++) {
        for (int h = 2; h <= max_h; h++) {
          double Ld = binomial(b + h - 2, h - 1);
          double Ls = binomial(b + h - 3, h - 1);

          // now we have k>=c*(1-alpha)^-2.
          // let's compute c.
          //double c = Math.log(2.0/(delta/quantiles)) / (2.0*epsilon*epsilon*Math.min(Ld, 8.0*Ls/3.0));
          double c = logDelta / Math.min(Ld, 8.0 * Ls / 3.0);

          // now we have k>=d/alpha.
          // let's compute d.
          double beta = Ld / Ls;
          double cc = (beta - 2.0) * (max_H - 2.0) / (beta + pow - 2.0);
          double d = (h + 3 + cc) / (2.0 * epsilon);

          /*
          double d = (Ld*(h+max_H-1.0)  +  Ls*((h+1)*pow - 2.0*(h+max_H)))   /   (Ld + Ls*(pow-2.0));
          d = (d + 2.0) / (2.0*epsilon);
          */

          // now we have c*(1-alpha)^-2 == d/alpha.
          // we solve this equation for alpha yielding two solutions
          // alpha_1,2 = (c + 2*d  +-  Sqrt(c*c + 4*c*d))/(2*d)
          double f = c * c + 4.0 * c * d;
          if (f < 0.0) {
            continue;
          } // non real solution to equation
          double root = Math.sqrt(f);
          double alpha_one = (c + 2.0 * d + root) / (2.0 * d);
          double alpha_two = (c + 2.0 * d - root) / (2.0 * d);

          // any alpha must satisfy 0<alpha<1 to yield valid solutions
          boolean alpha_one_OK = false;
          if (0.0 < alpha_one && alpha_one < 1.0) {
            alpha_one_OK = true;
          }
          boolean alpha_two_OK = false;
          if (0.0 < alpha_two && alpha_two < 1.0) {
            alpha_two_OK = true;
          }
          if (alpha_one_OK || alpha_two_OK) {
            double alpha = alpha_one;
            if (alpha_one_OK && alpha_two_OK) {
              // take the alpha that minimizes d/alpha
              alpha = Math.max(alpha_one, alpha_two);
            } else if (alpha_two_OK) {
              alpha = alpha_two;
            }

            // now we have k=Ceiling(Max(d/alpha, (h+1)/(2*epsilon)))
            long k = ceiling(Math.max(d / alpha, (h + 1) / (2.0 * epsilon)));
            if (k > 0) { // valid solution?
              long memory = b * k;
              if (memory < best_memory) {
                // found a solution requiring less memory
                best_k = k;
                best_b = b;
                best_h = h;
                best_memory = memory;
              }
            }
          }
        } //end for h
      } //end for b

      if (best_b == Long.MAX_VALUE) {
        log.info("Warning: Computing b and k looks like a lot of work!");
        // no solution found so far. very unlikely. Anyway, try again.
        max_b *= 2;
        max_h *= 2;
        max_H *= 2;
      }
    } //end while

    long[] result = new long[3];
    if (best_b == Long.MAX_VALUE) {
      // no solution found.
      // no way around exact quantile search.
      result[0] = 1;
      result[1] = Long.MAX_VALUE;
      result[2] = Long.MAX_VALUE;
    } else {
      result[0] = best_b;
      result[1] = best_k;
      result[2] = best_h;
    }

    return result;
  }
}
