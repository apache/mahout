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

import java.util.Random;

/** Partially deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
public final class Poisson extends AbstractDiscreteDistribution {

  private final double mean;

  // precomputed and cached values (for performance only)
  // cache for < SWITCH_MEAN
  private double myOld = -1.0;
  private double p;
  private double q;
  private double p0;
  private final double[] pp = new double[36];
  private int llll;

  // cache for >= SWITCH_MEAN
  private double myLast = -1.0;
  private double ll;
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
  private double lr;
  private double lMy;
  private double cPm;
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

  // cache for both;


  private static final double MEAN_MAX = Integer.MAX_VALUE;
  // for all means larger than that, we don't try to compute a poisson deviation, but return the mean.
  private static final double SWITCH_MEAN = 10.0; // switch from method A to method B


  /** Constructs a poisson distribution. Example: mean=1.0. */
  public Poisson(double mean, Random randomGenerator) {
    setRandomGenerator(randomGenerator);
    this.mean = mean;
  }

  private static double f(int k, double lNu, double cPm) {
    return Math.exp(k * lNu - Arithmetic.logFactorial(k) - cPm);
  }

  @Override
  public int nextInt() {
    return nextInt(mean);
  }

  /** Returns a random number from the distribution; bypasses the internal state. */
  public int nextInt(double theMean) {
    /******************************************************************
     *                                                                *
     * Poisson Distribution - Patchwork Rejection/Inversion           *
     *                                                                *
     ******************************************************************
     *                                                                *
     * For parameter  my < 10  Tabulated Inversion is applied.        *
     * For my >= 10  Patchwork Rejection is employed:                 *
     * The area below the histogram function f(x) is rearranged in    *
     * its body by certain point reflections. Within a large center   *
     * interval variates are sampled efficiently by rejection from    *
     * uniform hats. Rectangular immediate acceptance regions speed   *
     * up the generation. The remaining tails are covered by          *
     * exponential functions.                                         *
     *                                                                *
     *****************************************************************/
    Random gen = getRandomGenerator();

    //double t, g, my_k;

    //double gx, gy, px, py, e, x, xx, delta, v;
    //int sign;

    //static double p,q,p0,pp[36];
    //static long ll,m;

    int m;
    if (theMean < SWITCH_MEAN) { // CASE B: Inversion- start new table and calculate p0
      if (theMean != myOld) {
        myOld = theMean;
        llll = 0;
        p = Math.exp(-theMean);
        q = p;
        p0 = p;
        //for (k=pp.length; --k >=0;) pp[k] = 0;
      }
      m = theMean > 1.0 ? (int) theMean : 1;
      while (true) {
        double u = gen.nextDouble();
        int k = 0;
        if (u <= p0) {
          return k;
        }
        if (llll != 0) {              // Step T. Table comparison
          int i = u > 0.458 ? Math.min(llll, m) : 1;
          for (k = i; k <= llll; k++) {
            if (u <= pp[k]) {
              return k;
            }
          }
          if (llll == 35) {
            continue;
          }
        }
        for (k = llll + 1; k <= 35; k++) { // Step C. Creation of new prob.
          p *= theMean / k;
          q += p;
          pp[k] = q;
          if (u <= q) {
            llll = k;
            return k;
          }
        }
        llll = 35;
      }
      // end my < SWITCH_MEAN
    } else if (theMean < MEAN_MAX) { // CASE A: acceptance complement
      //static double        my_last = -1.0;
      //static long int      m,  k2, k4, k1, k5;
      //static double        dl, dr, r1, r2, r4, r5, ll, lr, l_my, c_pm,
      //             f1, f2, f4, f5, p1, p2, p3, p4, p5, p6;

      m = (int) theMean;
      if (theMean != myLast) { //  set-up
        myLast = theMean;

        // approximate deviation of reflection points k2, k4 from my - 1/2
        double Ds = Math.sqrt(theMean + 0.25);

        // mode m, reflection points k2 and k4, and points k1 and k5, which
        // delimit the centre region of h(x)
        k2 = (int) Math.ceil(theMean - 0.5 - Ds);
        k4 = (int) (theMean - 0.5 + Ds);
        k1 = k2 + k2 - m + 1;
        k5 = k4 + k4 - m;

        // range width of the critical left and right centre region
        dl = k2 - k1;
        dr = k5 - k4;

        // recurrence constants r(k) = p(k)/p(k-1) at k = k1, k2, k4+1, k5+1
        r1 = theMean / k1;
        r2 = theMean / k2;
        r4 = theMean / (k4 + 1);
        r5 = theMean / (k5 + 1);

        // reciprocal values of the scale parameters of expon. tail envelopes
        ll = Math.log(r1);                     // expon. tail left
        lr = -Math.log(r5);                     // expon. tail right

        // Poisson constants, necessary for computing function values f(k)
        lMy = Math.log(theMean);
        cPm = m * lMy - Arithmetic.logFactorial(m);

        // function values f(k) = p(k)/p(m) at k = k2, k4, k1, k5
        f2 = f(k2, lMy, cPm);
        f4 = f(k4, lMy, cPm);
        f1 = f(k1, lMy, cPm);
        f5 = f(k5, lMy, cPm);

        // area of the two centre and the two exponential tail regions
        // area of the two immediate acceptance regions between k2, k4
        p1 = f2 * (dl + 1.0);                    // immed. left
        p2 = f2 * dl + p1;               // centre left
        p3 = f4 * (dr + 1.0) + p2;               // immed. right
        p4 = f4 * dr + p3;               // centre right
        p5 = f1 / ll + p4;               // expon. tail left
        p6 = f5 / lr + p5;               // expon. tail right
      } // end set-up

      while (true) {
        // generate uniform number U -- U(0, p6)
        // case distinction corresponding to U
        double W;
        double V;
        double U;
        int Y;
        int X;
        int Dk;
        if ((U = gen.nextDouble() * p6) < p2) {         // centre left

          // immediate acceptance region R2 = [k2, m) *[0, f2),  X = k2, ... m -1
          if ((V = U - p1) < 0.0) {
            return k2 + (int) (U / f2);
          }
          // immediate acceptance region R1 = [k1, k2)*[0, f1),  X = k1, ... k2-1
          if ((W = V / dl) < f1) {
            return k1 + (int) (V / f1);
          }

          // computation of candidate X < k2, and its counterpart Y > k2
          // either squeeze-acceptance of X or acceptance-rejection of Y
          Dk = gen.nextInt((int) dl) + 1;
          if (W <= f2 - Dk * (f2 - f2 / r2)) {            // quick accept of
            return k2 - Dk;                          // X = k2 - Dk
          }
          if ((V = f2 + f2 - W) < 1.0) {                // quick reject of Y
            Y = k2 + Dk;
            if (V <= f2 + Dk * (1.0 - f2) / (dl + 1.0)) { // quick accept of
              return Y;                             // Y = k2 + Dk
            }
            if (V <= f(Y, lMy, cPm)) {
              return Y;
            }    // final accept of Y
          }
          X = k2 - Dk;
        } else if (U < p4) {                                 // centre right
          // immediate acceptance region R3 = [m, k4+1)*[0, f4), X = m, ... k4
          if ((V = U - p3) < 0.0) {
            return k4 - (int) ((U - p2) / f4);
          }
          // immediate acceptance region R4 = [k4+1, k5+1)*[0, f5)
          if ((W = V / dr) < f5) {
            return k5 - (int) (V / f5);
          }

          // computation of candidate X > k4, and its counterpart Y < k4
          // either squeeze-acceptance of X or acceptance-rejection of Y
          Dk = gen.nextInt((int) dr) + 1;
          if (W <= f4 - Dk * (f4 - f4 * r4)) {             // quick accept of
            return k4 + Dk;                           // X = k4 + Dk
          }
          if ((V = f4 + f4 - W) < 1.0) {                 // quick reject of Y
            Y = k4 - Dk;
            if (V <= f4 + Dk * (1.0 - f4) / dr) {       // quick accept of
              return Y;                             // Y = k4 - Dk
            }
            if (V <= f(Y, lMy, cPm)) {
              return Y;
            }    // final accept of Y
          }
          X = k4 + Dk;
        } else {
          W = gen.nextDouble();
          if (U < p5) {                                  // expon. tail left
            Dk = (int) (1.0 - Math.log(W) / ll);
            if ((X = k1 - Dk) < 0) {
              continue;
            }          // 0 <= X <= k1 - 1
            W *= (U - p4) * ll;                        // W -- U(0, h(x))
            if (W <= f1 - Dk * (f1 - f1 / r1)) {
              return X;
            } // quick accept of X
          } else {                                         // expon. tail right
            Dk = (int) (1.0 - Math.log(W) / lr);
            X = k5 + Dk;                              // X >= k5 + 1
            W *= (U - p5) * lr;                        // W -- U(0, h(x))
            if (W <= f5 - Dk * (f5 - f5 * r5)) {
              return X;
            } // quick accept of X
          }
        }

        // acceptance-rejection test of candidate X from the original area
        // test, whether  W <= f(k),    with  W = U*h(x)  and  U -- U(0, 1)
        // log f(X) = (X - m)*log(my) - log X! + log m!
        if (Math.log(W) <= X * lMy - Arithmetic.logFactorial(X) - cPm) {
          return X;
        }
      }
    } else { // mean is too large
      return (int) theMean;
    }
  }

}
