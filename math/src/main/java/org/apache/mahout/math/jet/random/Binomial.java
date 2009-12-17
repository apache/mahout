/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.random;

import org.apache.mahout.math.jet.math.Arithmetic;
import org.apache.mahout.math.jet.random.engine.RandomEngine;
import org.apache.mahout.math.jet.stat.Probability;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class Binomial extends AbstractDiscreteDistribution {

  private int n;
  private double p;

  // cache vars for method generateBinomial(...)
  private int n_last = -1, n_prev = -1;
  private double par, np, p0, q, p_last = -1.0, p_prev = -1.0;
  private int b, m, nm;
  private double pq, rc, ss, xm, xl, xr, ll, lr, c, p1, p2, p3, p4, ch;

  // cache vars for method pdf(...)
  private double log_p, log_q, log_n;

  // The uniform random number generated shared by all <b>static</b> methods.
  private static final Binomial shared = new Binomial(1, 0.5, makeDefaultGenerator());

  /**
   * Constructs a binomial distribution. Example: n=1, p=0.5.
   *
   * @param n               the number of trials (also known as <i>sample size</i>).
   * @param p               the probability of success.
   * @param randomGenerator a uniform random number generator.
   * @throws IllegalArgumentException if <tt>n*Math.min(p,1-p) &lt;= 0.0</tt>
   */
  public Binomial(int n, double p, RandomEngine randomGenerator) {
    setRandomGenerator(randomGenerator);
    setNandP(n, p);
  }

  /** Returns the cumulative distribution function. */
  public double cdf(int k) {
    return Probability.binomial(k, n, p);
  }

  /** Returns the cumulative distribution function. */
  private double cdfSlow(int k) {
    if (k < 0) {
      throw new IllegalArgumentException();
    }

    double sum = 0.0;
    for (int r = 0; r <= k; r++) {
      sum += pdf(r);
    }

    return sum;
  }

  /**
   * *************************************************************** * Binomial-Distribution - Acceptance
   * Rejection/Inversion     * * ***************************************************************** * Acceptance
   * Rejection method combined with Inversion for        * generating Binomial random numbers with parameters * n
   * (number of trials) and p (probability of success).           * For  min(n*p,n*(1-p)) < 10  the Inversion method is
   * applied:   * The random numbers are generated via sequential search,        * starting at the lowest index k=0. The
   * cumulative probabilities * are avoided by using the technique of chop-down.               * For min(n*p,n*(1-p)) >=
   * 10  Acceptance Rejection is used:     * The algorithm is based on a hat-function which is uniform in   * the centre
   * region and exponential in the tails.                * A triangular immediate acceptance region in the centre speeds
   * * up the generation of binomial variates.                        * If candidate k is near the mode, f(k) is
   * computed recursively  * starting at the mode m.                                        * The acceptance test by
   * Stirling's formula is modified          * according to W. Hoermann (1992): The generation of binomial    * random
   * variates, to appear in J. Statist. Comput. Simul.       * If  p < .5  the algorithm is applied to parameters n, p.
   * * Otherwise p is replaced by 1-p, and k is replaced by n - k.    * * *****************************************************************
   * * FUNCTION:    - samples a random number from the binomial       * distribution with parameters n and p  and is *
   * valid for  n*min(p,1-p)  >  0. * REFERENCE:   - V. Kachitvichyanukul, B.W. Schmeiser (1988):    * Binomial random
   * variate generation, * Communications of the ACM 31, 216-222.          * SUBPROGRAMS: - StirlingCorrection() * ...
   * Correction term of the Stirling * approximation for log(k!) * (series in 1/k or table values  * for small k) with
   * long int k    * - randomGenerator    ... (0,1)-Uniform engine * * ****************************************************************
   */
  protected int generateBinomial(int n, double p) {


    int i;
    double f;

    if (n != n_last || p != p_last) {                 // set-up
      n_last = n;
      p_last = p;
      par = Math.min(p, 1.0 - p);
      q = 1.0 - par;
      np = n * par;

      // Check for invalid input values

      if (np <= 0.0) {
        return -1;
      }

      double rm = np + par;
      m = (int) rm;                            // mode, integer
      if (np < 10) {
        p0 = Math.exp(n * Math.log(q));               // Chop-down
        int bh = (int) (np + 10.0 * Math.sqrt(np * q));
        b = Math.min(n, bh);
      } else {
        rc = (n + 1.0) * (pq = par / q);          // recurr. relat.
        ss = np * q;                              // variance
        i = (int) (2.195 * Math.sqrt(ss) - 4.6 * q); // i = p1 - 0.5
        xm = m + 0.5;
        xl = (double) (m - i);                    // limit left
        xr = (double) (m + i + 1L);               // limit right
        f = (rm - xl) / (rm - xl * par);
        ll = f * (1.0 + 0.5 * f);
        f = (xr - rm) / (xr * q);
        lr = f * (1.0 + 0.5 * f);
        c = 0.134 + 20.5 / (15.3 + (double) m);    // parallelogram
        // height
        p1 = i + 0.5;
        p2 = p1 * (1.0 + c + c);                  // probabilities
        p3 = p2 + c / ll;                           // of regions 1-4
        p4 = p3 + c / lr;
      }
    }

    double U;
    int K;
    if (np < 10) {                                      //Inversion Chop-down

      K = 0;
      double pk = p0;
      U = randomGenerator.raw();
      while (U > pk) {
        ++K;
        if (K > b) {
          U = randomGenerator.raw();
          K = 0;
          pk = p0;
        } else {
          U -= pk;
          pk = ((n - K + 1) * par * pk) / (K * q);
        }
      }
      return ((p > 0.5) ? (n - K) : K);
    }

    int DMAX_KM = 20;
    double C1_6 = 0.16666666666666667;
    double C5_8 = 0.62500000000000000;
    double C1_3 = 0.33333333333333333;
    while (true) {
      double V = randomGenerator.raw();
      if ((U = randomGenerator.raw() * p4) <= p1) {    // triangular region
        K = (int) (xm - U + p1 * V);
        return (p > 0.5) ? (n - K) : K;  // immediate accept
      }
      double X;
      if (U <= p2) {                                  // parallelogram
        X = xl + (U - p1) / c;
        if ((V = V * c + 1.0 - Math.abs(xm - X) / p1) >= 1.0) {
          continue;
        }
        K = (int) X;
      } else if (U <= p3) {                              // left tail
        if ((X = xl + Math.log(V) / ll) < 0.0) {
          continue;
        }
        K = (int) X;
        V *= (U - p2) * ll;
      } else {                                           // right tail
        if ((K = (int) (xr - Math.log(V) / lr)) > n) {
          continue;
        }
        V *= (U - p3) * lr;
      }

      // acceptance test :  two cases, depending on |K - m|
      int Km;
      if ((Km = Math.abs(K - m)) <= DMAX_KM || Km + Km + 2L >= ss) {

        // computation of p(K) via recurrence relationship from the mode
        f = 1.0;                              // f(m)
        if (m < K) {
          for (i = m; i < K;) {
            if ((f *= (rc / ++i - pq)) < V) {
              break;
            }  // multiply  f
          }
        } else {
          for (i = K; i < m;) {
            if ((V *= (rc / ++i - pq)) > f) {
              break;
            }  // multiply  V
          }
        }
        if (V <= f) {
          break;
        }                            // acceptance test
      } else {

        // lower and upper squeeze tests, based on lower bounds for log p(K)
        V = Math.log(V);
        double T = -Km * Km / (ss + ss);
        double E = (Km / ss) * ((Km * (Km * C1_3 + C5_8) + C1_6) / ss + 0.5);
        if (V <= T - E) {
          break;
        }
        if (V <= T + E) {
          if (n != n_prev || par != p_prev) {
            n_prev = n;
            p_prev = par;

            nm = n - m + 1;
            ch = xm * Math.log((m + 1.0) / (pq * nm)) +
                Arithmetic.stirlingCorrection(m + 1) + Arithmetic.stirlingCorrection(nm);
          }
          int nK = n - K + 1;

          // computation of log f(K) via Stirling's formula
          // final acceptance-rejection test
          if (V <= ch + (n + 1.0) * Math.log((double) nm / (double) nK) +
              (K + 0.5) * Math.log(nK * pq / (K + 1.0)) -
              Arithmetic.stirlingCorrection(K + 1) - Arithmetic.stirlingCorrection(nK)) {
            break;
          }
        }
      }
    }
    return (p > 0.5) ? (n - K) : K;
  }

  /** Returns a random number from the distribution. */
  @Override
  public int nextInt() {
    return generateBinomial(n, p);
  }

  /**
   * Returns a random number from the distribution with the given parameters n and p; bypasses the internal state.
   *
   * @param n the number of trials
   * @param p the probability of success.
   * @throws IllegalArgumentException if <tt>n*Math.min(p,1-p) &lt;= 0.0</tt>
   */
  public int nextInt(int n, double p) {
    if (n * Math.min(p, 1 - p) <= 0.0) {
      throw new IllegalArgumentException();
    }
    return generateBinomial(n, p);
  }

  /** Returns the probability distribution function. */
  public double pdf(int k) {
    if (k < 0) {
      throw new IllegalArgumentException();
    }
    int r = this.n - k;
    return Math
        .exp(this.log_n - Arithmetic.logFactorial(k) - Arithmetic.logFactorial(r) + this.log_p * k + this.log_q * r);
  }

  /**
   * Sets the parameters number of trials and the probability of success.
   *
   * @param n the number of trials
   * @param p the probability of success.
   * @throws IllegalArgumentException if <tt>n*Math.min(p,1-p) &lt;= 0.0</tt>
   */
  public void setNandP(int n, double p) {
    if (n * Math.min(p, 1 - p) <= 0.0) {
      throw new IllegalArgumentException();
    }
    this.n = n;
    this.p = p;

    this.log_p = Math.log(p);
    this.log_q = Math.log(1.0 - p);
    this.log_n = Arithmetic.logFactorial(n);
  }

  /**
   * Returns a random number from the distribution with the given parameters n and p.
   *
   * @param n the number of trials
   * @param p the probability of success.
   * @throws IllegalArgumentException if <tt>n*Math.min(p,1-p) &lt;= 0.0</tt>
   */
  public static int staticNextInt(int n, double p) {
    synchronized (shared) {
      return shared.nextInt(n, p);
    }
  }

  /** Returns a String representation of the receiver. */
  public String toString() {
    return this.getClass().getName() + '(' + n + ',' + p + ')';
  }

}
