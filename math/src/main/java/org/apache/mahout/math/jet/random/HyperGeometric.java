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

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class HyperGeometric extends AbstractDiscreteDistribution {

  private int my_N;
  private int my_s;
  private int my_n;

  // cached vars shared by hmdu(...) and hprs(...)
  private int N_last = -1;
  private int M_last = -1;
  private int n_last = -1;
  private int N_Mn;
  private int m;

  // cached vars for hmdu(...)
  private int mp;
  private int b;
  private double Mp;
  private double np;
  private double fm;

  // cached vars for hprs(...)
  private int k2;
  private int k4;
  private int k1;
  private int k5;
  private double dl;
  private double dr;
  private double r1;
  private double r2;
  private double r4;
  private double r5;
  private double ll;
  private double lr;
  private double c_pm;
  private double f1;
  private double f2;
  private double f4;
  private double f5;
  private double p1;
  private double p2;
  private double p3;
  private double p4;
  private double p5;
  private double p6;


  // The uniform random number generated shared by all <b>static</b> methods.
  private static final HyperGeometric shared = new HyperGeometric(1, 1, 1, makeDefaultGenerator());

  /** Constructs a HyperGeometric distribution. */
  public HyperGeometric(int N, int s, int n, RandomEngine randomGenerator) {
    setRandomGenerator(randomGenerator);
    setState(N, s, n);
  }

  private static double fc_lnpk(int k, int N_Mn, int M, int n) {
    return (Arithmetic.logFactorial(k) + Arithmetic.logFactorial(M - k) + Arithmetic.logFactorial(n - k) +
        Arithmetic.logFactorial(N_Mn + k));
  }

  /** Returns a random number from the distribution. */
  protected int hmdu(int N, int M, int n, RandomEngine randomGenerator) {

    if (N != N_last || M != M_last || n != n_last) {   // set-up           */
      N_last = N;
      M_last = M;
      n_last = n;

      Mp = (double) (M + 1);
      np = (double) (n + 1);
      N_Mn = N - M - n;

      double p = Mp / (N + 2.0);
      double nu = np * p;
      if ((m = (int) nu) == nu && p == 0.5) {     /* mode, integer    */
        mp = m--;
      } else {
        mp = m + 1;                           /* mp = m + 1       */
      }

      /* mode probability, using the external function flogfak(k) = ln(k!)    */
      fm = Math.exp(Arithmetic.logFactorial(N - M) - Arithmetic.logFactorial(N_Mn + m) - Arithmetic.logFactorial(n - m)
          + Arithmetic.logFactorial(M) - Arithmetic.logFactorial(M - m) - Arithmetic.logFactorial(m)
          - Arithmetic.logFactorial(N) + Arithmetic.logFactorial(N - n) + Arithmetic.logFactorial(n));

      /* safety bound  -  guarantees at least 17 significant decimal digits   */
      /*                  b = min(n, (long int)(nu + k*c')) */
      b = (int) (nu + 11.0 * Math.sqrt(nu * (1.0 - p) * (1.0 - n / (double) N) + 1.0));
      if (b > n) {
        b = n;
      }
    }

    while (true) {
      double U;
      if ((U = randomGenerator.raw() - fm) <= 0.0) {
        return (m);
      }
      double d;
      double c = d = fm;

      /* down- and upward search from the mode                                */
      int K;
      for (int I = 1; I <= m; I++) {
        K = mp - I;                                   /* downward search  */
        c *= (double) K / (np - K) * ((double) (N_Mn + K) / (Mp - K));
        if ((U -= c) <= 0.0) {
          return (K - 1);
        }

        K = m + I;                                    /* upward search    */
        d *= (np - K) / (double) K * ((Mp - K) / (double) (N_Mn + K));
        if ((U -= d) <= 0.0) {
          return (K);
        }
      }

      /* upward search from K = 2m + 1 to K = b                               */
      for (K = mp + m; K <= b; K++) {
        d *= (np - K) / (double) K * ((Mp - K) / (double) (N_Mn + K));
        if ((U -= d) <= 0.0) {
          return (K);
        }
      }
    }
  }

  /** Returns a random number from the distribution. */
  protected int hprs(int N, int M, int n, RandomEngine randomGenerator) {
    double U;         /* (X, Y) <-> (V, W) */

    if (N != N_last || M != M_last || n != n_last) {  /* set-up            */
      N_last = N;
      M_last = M;
      n_last = n;

      double Mp = (double) (M + 1);
      double np = (double) (n + 1);
      N_Mn = N - M - n;

      double p = Mp / (N + 2.0);
      double nu = np * p;

      // approximate deviation of reflection points k2, k4 from nu - 1/2
      U = Math.sqrt(nu * (1.0 - p) * (1.0 - (n + 2.0) / (N + 3.0)) + 0.25);

      // mode m, reflection points k2 and k4, and points k1 and k5, which
      // delimit the centre region of h(x)
      // k2 = ceil (nu - 1/2 - U),    k1 = 2*k2 - (m - 1 + delta_ml)
      // k4 = floor(nu - 1/2 + U),    k5 = 2*k4 - (m + 1 - delta_mr)

      m = (int) nu;
      k2 = (int) Math.ceil(nu - 0.5 - U);
      if (k2 >= m) {
        k2 = m - 1;
      }
      k4 = (int) (nu - 0.5 + U);
      k1 = k2 + k2 - m + 1;                    // delta_ml = 0
      k5 = k4 + k4 - m;                               // delta_mr = 1

      // range width of the critical left and right centre region
      dl = (double) (k2 - k1);
      dr = (double) (k5 - k4);

      // recurrence constants r(k) = p(k)/p(k-1) at k = k1, k2, k4+1, k5+1
      r1 = (np / (double) k1 - 1.0) * (Mp - k1) / (double) (N_Mn + k1);
      r2 = (np / (double) k2 - 1.0) * (Mp - k2) / (double) (N_Mn + k2);
      r4 = (np / (double) (k4 + 1) - 1.0) * (M - k4) / (double) (N_Mn + k4 + 1);
      r5 = (np / (double) (k5 + 1) - 1.0) * (M - k5) / (double) (N_Mn + k5 + 1);

      // reciprocal values of the scale parameters of expon. tail envelopes
      ll = Math.log(r1);                                  // expon. tail left  //
      lr = -Math.log(r5);                                  // expon. tail right //

      // hypergeom. constant, necessary for computing function values f(k)
      c_pm = fc_lnpk(m, N_Mn, M, n);

      // function values f(k) = p(k)/p(m)  at  k = k2, k4, k1, k5
      f2 = Math.exp(c_pm - fc_lnpk(k2, N_Mn, M, n));
      f4 = Math.exp(c_pm - fc_lnpk(k4, N_Mn, M, n));
      f1 = Math.exp(c_pm - fc_lnpk(k1, N_Mn, M, n));
      f5 = Math.exp(c_pm - fc_lnpk(k5, N_Mn, M, n));

      // area of the two centre and the two exponential tail regions
      // area of the two immediate acceptance regions between k2, k4
      p1 = f2 * (dl + 1.0);                           // immed. left
      p2 = f2 * dl + p1;                      // centre left
      p3 = f4 * (dr + 1.0) + p2;                      // immed. right
      p4 = f4 * dr + p3;                      // centre right
      p5 = f1 / ll + p4;                      // expon. tail left
      p6 = f5 / lr + p5;                      // expon. tail right
    }

    while (true) {
      // generate uniform number U -- U(0, p6)
      // case distinction corresponding to U
      double W;
      double Y;
      int V;
      int X;
      int Dk;
      if ((U = randomGenerator.raw() * p6) < p2) {    // centre left

        // immediate acceptance region R2 = [k2, m) *[0, f2),  X = k2, ... m -1
        if ((W = U - p1) < 0.0) {
          return (k2 + (int) (U / f2));
        }
        // immediate acceptance region R1 = [k1, k2)*[0, f1),  X = k1, ... k2-1
        if ((Y = W / dl) < f1) {
          return (k1 + (int) (W / f1));
        }

        // computation of candidate X < k2, and its counterpart V > k2
        // either squeeze-acceptance of X or acceptance-rejection of V
        Dk = (int) (dl * randomGenerator.raw()) + 1;
        if (Y <= f2 - Dk * (f2 - f2 / r2)) {            // quick accept of
          return (k2 - Dk);                          // X = k2 - Dk
        }
        if ((W = f2 + f2 - Y) < 1.0) {                // quick reject of V
          V = k2 + Dk;
          if (W <= f2 + Dk * (1.0 - f2) / (dl + 1.0)) { // quick accept of
            return (V);                              // V = k2 + Dk
          }
          if (Math.log(W) <= c_pm - fc_lnpk(V, N_Mn, M, n)) {
            return (V);               // final accept of V
          }
        }
        X = k2 - Dk;
      } else if (U < p4) {                              // centre right

        // immediate acceptance region R3 = [m, k4+1)*[0, f4), X = m, ... k4
        if ((W = U - p3) < 0.0) {
          return (k4 - (int) ((U - p2) / f4));
        }
        // immediate acceptance region R4 = [k4+1, k5+1)*[0, f5)
        if ((Y = W / dr) < f5) {
          return (k5 - (int) (W / f5));
        }

        // computation of candidate X > k4, and its counterpart V < k4
        // either squeeze-acceptance of X or acceptance-rejection of V
        Dk = (int) (dr * randomGenerator.raw()) + 1;
        if (Y <= f4 - Dk * (f4 - f4 * r4)) {            // quick accept of
          return (k4 + Dk);                          // X = k4 + Dk
        }
        if ((W = f4 + f4 - Y) < 1.0) {                // quick reject of V
          V = k4 - Dk;
          if (W <= f4 + Dk * (1.0 - f4) / dr) {       // quick accept of
            return (V);                            // V = k4 - Dk
          }
          if (Math.log(W) <= c_pm - fc_lnpk(V, N_Mn, M, n)) {
            return (V);                            // final accept of V
          }
        }
        X = k4 + Dk;
      } else {
        Y = randomGenerator.raw();
        if (U < p5) {                                 // expon. tail left
          Dk = (int) (1.0 - Math.log(Y) / ll);
          if ((X = k1 - Dk) < 0) {
            continue;
          }         // 0 <= X <= k1 - 1
          Y *= (U - p4) * ll;                       // Y -- U(0, h(x))
          if (Y <= f1 - Dk * (f1 - f1 / r1)) {
            return (X);                            // quick accept of X
          }
        } else {                                        // expon. tail right
          Dk = (int) (1.0 - Math.log(Y) / lr);
          if ((X = k5 + Dk) > n) {
            continue;
          }        // k5 + 1 <= X <= n
          Y *= (U - p5) * lr;                       // Y -- U(0, h(x))   /
          if (Y <= f5 - Dk * (f5 - f5 * r5)) {
            return (X);                            // quick accept of X
          }
        }
      }

      // acceptance-rejection test of candidate X from the original area
      // test, whether  Y <= f(X),    with  Y = U*h(x)  and  U -- U(0, 1)
      // log f(X) = log( m! (M - m)! (n - m)! (N - M - n + m)! )
      //          - log( X! (M - X)! (n - X)! (N - M - n + X)! )
      // by using an external function for log k!
      if (Math.log(Y) <= c_pm - fc_lnpk(X, N_Mn, M, n)) {
        return (X);
      }
    }
  }

  /** Returns a random number from the distribution. */
  @Override
  public int nextInt() {
    return nextInt(this.my_N, this.my_s, this.my_n, this.randomGenerator);
  }

  /** Returns a random number from the distribution; bypasses the internal state. */
  public int nextInt(int N, int s, int n) {
    return nextInt(N, s, n, this.randomGenerator);
  }

  /** Returns a random number from the distribution; bypasses the internal state. */
  protected int nextInt(int N, int M, int n, RandomEngine randomGenerator) {
/******************************************************************
 *                                                                *
 * Hypergeometric Distribution - Patchwork Rejection/Inversion    *
 *                                                                *
 ******************************************************************
 *                                                                *
 * The basic algorithms work for parameters 1 <= n <= M <= N/2.   *
 * Otherwise parameters are re-defined in the set-up step and the *
 * random number K is adapted before delivering.                  *
 * For l = m-max(0,n-N+M) < 10  Inversion method hmdu is applied: *
 * The random numbers are generated via modal down-up search,     *
 * starting at the mode m. The cumulative probabilities           *
 * are avoided by using the technique of chop-down.               *
 * For l >= 10  the Patchwork Rejection method  hprs is employed: *
 * The area below the histogram function f(x) in its              *
 * body is rearranged by certain point reflections. Within a      *
 * large center interval variates are sampled efficiently by      *
 * rejection from uniform hats. Rectangular immediate acceptance  *
 * regions speed up the generation. The remaining tails are       *
 * covered by exponential functions.                              *
 *                                                                *
 ******************************************************************
 *                                                                *
 * FUNCTION :   - hprsc samples a random number from the          *
 *                Hypergeometric distribution with parameters     *
 *                N (number of red and black balls), M (number    *
 *                of red balls) and n (number of trials)          *
 *                valid for N >= 2, M,n <= N.                     *
 * REFERENCE :  - H. Zechner (1994): Efficient sampling from      *
 *                continuous and discrete unimodal distributions, *
 *                Doctoral Dissertation, 156 pp., Technical       *
 *                University Graz, Austria.                       *
 * SUBPROGRAMS: - flogfak(k)  ... log(k!) with long integer k     *
 *              - drand(seed) ... (0,1)-Uniform generator with    *
 *                unsigned long integer *seed.                    *
 *              - hmdu(seed,N,M,n) ... Hypergeometric generator   *
 *                for l<10                                        *
 *              - hprs(seed,N,M,n) ... Hypergeometric generator   *
 *                for l>=10 with unsigned long integer *seed,     *
 *                long integer  N , M , n.                        *
 *                                                                *
 ******************************************************************/

    int Nhalf = N / 2;
    int n_le_Nhalf = (n <= Nhalf) ? n : N - n;
    int M_le_Nhalf = (M <= Nhalf) ? M : N - M;

    int K;
    if ((n * M / N) < 10) {
      K = (n_le_Nhalf <= M_le_Nhalf)
          ? hmdu(N, M_le_Nhalf, n_le_Nhalf, randomGenerator)
          : hmdu(N, n_le_Nhalf, M_le_Nhalf, randomGenerator);
    } else {
      K = (n_le_Nhalf <= M_le_Nhalf)
          ? hprs(N, M_le_Nhalf, n_le_Nhalf, randomGenerator)
          : hprs(N, n_le_Nhalf, M_le_Nhalf, randomGenerator);
    }

    if (n <= Nhalf) {
      return (M <= Nhalf) ? K : n - K;
    } else {
      return (M <= Nhalf) ? M - K : n - N + M + K;
    }
  }

  /** Returns the probability distribution function. */
  public double pdf(int k) {
    return Arithmetic.binomial(my_s, k) * Arithmetic.binomial(my_N - my_s, my_n - k)
        / Arithmetic.binomial(my_N, my_n);
  }

  /** Sets the parameters. */
  public void setState(int N, int s, int n) {
    this.my_N = N;
    this.my_s = s;
    this.my_n = n;
  }

  /** Returns a random number from the distribution. */
  public static double staticNextInt(int N, int M, int n) {
    synchronized (shared) {
      return shared.nextInt(N, M, n);
    }
  }

  /** Returns a String representation of the receiver. */
  public String toString() {
    return this.getClass().getName() + '(' + my_N + ',' + my_s + ',' + my_n + ')';
  }

}
