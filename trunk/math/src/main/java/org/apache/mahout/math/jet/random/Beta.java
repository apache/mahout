/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.random;

import org.apache.mahout.math.jet.stat.Probability;

import java.util.Random;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class Beta extends AbstractContinousDistribution {

  private double alpha;
  private double beta;

  private double pdfConst; // cache to speed up pdf()

  // cached values shared by bXX
  private double aLast;
  private double bLast;
  private double aMinus1;
  private double bMinus1;
  private double t;
  private double fa;
  private double fb;
  private double p1;
  private double p2;

  // cached values for b00

  // cached values for b01
  private double ml;
  private double mu;

  // cached values for b1prs
  private double pLast;
  private double qLast;
  private double a;
  private double b;
  private double m;
  private double d;
  private double d1;
  private double x1;
  private double x2;
  private double x4;
  private double x5;
  private double f1;
  private double f2;
  private double f4;
  private double f5;
  private double ll;
  private double lr;
  private double z2;
  private double z4;
  private double p3;
  private double p4;

  /** Constructs a Beta distribution. */
  public Beta(double alpha, double beta, Random randomGenerator) {
    setRandomGenerator(randomGenerator);
    setState(alpha, beta);
  }

  /**
   *
   */
  protected double b00(double a, double b, Random randomGenerator) {

    if (a != aLast || b != bLast) {
      aLast = a;
      bLast = b;

      aMinus1 = a - 1.0;
      bMinus1 = b - 1.0;
      double c = (b * bMinus1) / (a * aMinus1);
      t = Math.abs(c - 1.0) < 1.0e-8 ? 0.5 : (1.0 - Math.sqrt(c)) / (1.0 - c);
      fa = Math.exp(aMinus1 * Math.log(t));
      fb = Math.exp(bMinus1 * Math.log(1.0 - t));              // f(t) = fa * fb

      p1 = t / a;                                           // 0 < X < t
      p2 = (1.0 - t) / b + p1;                              // t < X < 1
    }

    double X;
    while (true) {
      double Z;
      double V;
      double U;
      if ((U = randomGenerator.nextDouble() * p2) <= p1) {       //  X < t
        Z = Math.exp(Math.log(U / p1) / a);
        X = t * Z;
        // squeeze accept:   L(x) = 1 + (1 - b)x
        if ((V = randomGenerator.nextDouble() * fb) <= 1.0 - bMinus1 * X) {
          break;
        }
        // squeeze reject:   U(x) = 1 + ((1 - t)^(b-1) - 1)/t * x
        if (V <= 1.0 + (fb - 1.0) * Z) {
          // quotient accept:  q(x) = (1 - x)^(b-1) / fb
          if (Math.log(V) <= bMinus1 * Math.log(1.0 - X)) {
            break;
          }
        }
      } else {                                                      //  X > t
        Z = Math.exp(Math.log((U - p1) / (p2 - p1)) / b);
        X = 1.0 - (1.0 - t) * Z;
        // squeeze accept:   L(x) = 1 + (1 - a)(1 - x)
        if ((V = randomGenerator.nextDouble() * fa) <= 1.0 - aMinus1 * (1.0 - X)) {
          break;
        }
        // squeeze reject:   U(x) = 1 + (t^(a-1) - 1)/(1 - t) * (1 - x)
        if (V <= 1.0 + (fa - 1.0) * Z) {
          // quotient accept:  q(x) = x^(a-1) / fa
          if (Math.log(V) <= aMinus1 * Math.log(X)) {
            break;
          }
        }
      }
    }
    return X;
  }

  protected double b01(double a, double b, Random randomGenerator) {

    if (a != aLast || b != bLast) {
      aLast = a;
      bLast = b;

      aMinus1 = a - 1.0;
      bMinus1 = b - 1.0;
      t = aMinus1 / (a - b);                   // one step Newton * start value t
      fb = Math.exp((bMinus1 - 1.0) * Math.log(1.0 - t));
      fa = a - (a + bMinus1) * t;
      t -= (t - (1.0 - fa) * (1.0 - t) * fb / b) / (1.0 - fa * fb);
      fa = Math.exp(aMinus1 * Math.log(t));
      fb = Math.exp(bMinus1 * Math.log(1.0 - t));             // f(t) = fa * fb
      if (bMinus1 <= 1.0) {
        ml = (1.0 - fb) / t;                           //   ml = -m1
        mu = bMinus1 * t;                                   //   mu = -m2 * t
      } else {
        ml = bMinus1;
        mu = 1.0 - fb;
      }
      p1 = t / a;                                           //  0 < X < t
      p2 = fb * (1.0 - t) / b + p1;                         //  t < X < 1
    }

    double X;
    while (true) {
      double Z;
      double V;
      double U;
      if ((U = randomGenerator.nextDouble() * p2) <= p1) {       //  X < t
        Z = Math.exp(Math.log(U / p1) / a);
        X = t * Z;
        // squeeze accept:   L(x) = 1 + m1*x,  ml = -m1
        if ((V = randomGenerator.nextDouble()) <= 1.0 - ml * X) {
          break;
        }
        // squeeze reject:   U(x) = 1 + m2*x,  mu = -m2 * t
        if (V <= 1.0 - mu * Z) {
          // quotient accept:  q(x) = (1 - x)^(b-1)
          if (Math.log(V) <= bMinus1 * Math.log(1.0 - X)) {
            break;
          }
        }
      } else {                                                      //  X > t
        Z = Math.exp(Math.log((U - p1) / (p2 - p1)) / b);
        X = 1.0 - (1.0 - t) * Z;
        // squeeze accept:   L(x) = 1 + (1 - a)(1 - x)
        V = randomGenerator.nextDouble() * fa;
        if (V <= 1.0 - aMinus1 * (1.0 - X)) {
          break;
        }
        // squeeze reject:   U(x) = 1 + (t^(a-1) - 1)/(1 - t) * (1 - x)
        if (V <= 1.0 + (fa - 1.0) * Z) {
          // quotient accept:  q(x) = (x)^(a-1) / fa
          if (Math.log(V) <= aMinus1 * Math.log(X)) {
            break;
          }
        }
      }
    }
    return X;
  }

  protected double b1prs(double p, double q, Random randomGenerator) {

    if (p != pLast || q != qLast) {
      pLast = p;
      qLast = q;

      a = p - 1.0;
      b = q - 1.0;
      double s = a + b;
      m = a / s;
      if (a > 1.0 || b > 1.0) {
        d = Math.sqrt(m * (1.0 - m) / (s - 1.0));
      }

      if (a <= 1.0) {
        d1 = m * 0.5;
        x2 = d1;
        x1 = z2 = 0.0;
        f1 = ll = 0.0;
      } else {
        x2 = m - d;
        x1 = x2 - d;
        z2 = x2 * (1.0 - (1.0 - x2) / (s * d));
        if (x1 <= 0.0 || (s - 6.0) * x2 - a + 3.0 > 0.0) {
          x1 = z2;
          x2 = (x1 + m) * 0.5;
          d1 = m - x2;
        } else {
          d1 = d;
        }
        f1 = f(x1, a, b, m);
        ll = x1 * (1.0 - x1) / (s * (m - x1));          // z1 = x1 - ll
      }
      f2 = f(x2, a, b, m);

      if (b <= 1.0) {
        x4 = 1.0 - (d = (1.0 - m) * 0.5);
        x5 = z4 = 1.0;
        f5 = lr = 0.0;
      } else {
        x4 = m + d;
        x5 = x4 + d;
        z4 = x4 * (1.0 + (1.0 - x4) / (s * d));
        if (x5 >= 1.0 || (s - 6.0) * x4 - a + 3.0 < 0.0) {
          x5 = z4;
          x4 = (m + x5) * 0.5;
          d = x4 - m;
        }
        f5 = f(x5, a, b, m);
        lr = x5 * (1.0 - x5) / (s * (x5 - m));          // z5 = x5 + lr
      }
      f4 = f(x4, a, b, m);

      p1 = f2 * (d1 + d1);                                //  x1 < X < m
      p2 = f4 * (d + d) + p1;                            //  m  < X < x5
      p3 = f1 * ll + p2;                            //       X < x1
      p4 = f5 * lr + p3;                            //  x5 < X
    }

    while (true) {
      double Y;
      double X;
      double W;
      double V;
      double U;
      if ((U = randomGenerator.nextDouble() * p4) <= p1) {
        // immediate accept:  x2 < X < m, - f(x2) < W < 0
        if ((W = U / d1 - f2) <= 0.0) {
          return m - U / f2;
        }
        // immediate accept:  x1 < X < x2, 0 < W < f(x1)
        if (W <= f1) {
          return x2 - W / f1 * d1;
        }
        // candidates for acceptance-rejection-test
        V = d1 * (U = randomGenerator.nextDouble());
        X = x2 - V;
        Y = x2 + V;
        // squeeze accept:    L(x) = f(x2) (x - z2) / (x2 - z2)
        if (W * (x2 - z2) <= f2 * (X - z2)) {
          return X;
        }
        if ((V = f2 + f2 - W) < 1.0) {
          // squeeze accept:    L(x) = f(x2) + (1 - f(x2))(x - x2)/(m - x2)
          if (V <= f2 + (1.0 - f2) * U) {
            return Y;
          }
          // quotient accept:   x2 < Y < m,   W >= 2f2 - f(Y)
          if (V <= f(Y, a, b, m)) {
            return Y;
          }
        }
      } else if (U <= p2) {
        U -= p1;
        // immediate accept:  m < X < x4, - f(x4) < W < 0
        if ((W = U / d - f4) <= 0.0) {
          return m + U / f4;
        }
        // immediate accept:  x4 < X < x5, 0 < W < f(x5)
        if (W <= f5) {
          return x4 + W / f5 * d;
        }
        // candidates for acceptance-rejection-test
        V = d * (U = randomGenerator.nextDouble());
        X = x4 + V;
        Y = x4 - V;
        // squeeze accept:    L(x) = f(x4) (z4 - x) / (z4 - x4)
        if (W * (z4 - x4) <= f4 * (z4 - X)) {
          return X;
        }
        if ((V = f4 + f4 - W) < 1.0) {
          // squeeze accept:    L(x) = f(x4) + (1 - f(x4))(x4 - x)/(x4 - m)
          if (V <= f4 + (1.0 - f4) * U) {
            return Y;
          }
          // quotient accept:   m < Y < x4,   W >= 2f4 - f(Y)
          if (V <= f(Y, a, b, m)) {
            return Y;
          }
        }
      } else if (U <= p3) {                                     // X < x1
        Y = Math.log(U = (U - p2) / (p3 - p2));
        if ((X = x1 + ll * Y) <= 0.0) {
          continue;
        }            // X > 0!!
        W = randomGenerator.nextDouble() * U;
        // squeeze accept:    L(x) = f(x1) (x - z1) / (x1 - z1)
        //                    z1 = x1 - ll,   W <= 1 + (X - x1)/ll
        if (W <= 1.0 + Y) {
          return X;
        }
        W *= f1;
      } else {                                                  // x5 < X
        Y = Math.log(U = (U - p3) / (p4 - p3));
        if ((X = x5 - lr * Y) >= 1.0) {
          continue;
        }            // X < 1!!
        W = randomGenerator.nextDouble() * U;
        // squeeze accept:    L(x) = f(x5) (z5 - x) / (z5 - x5)
        //                    z5 = x5 + lr,   W <= 1 + (x5 - X)/lr
        if (W <= 1.0 + Y) {
          return X;
        }
        W *= f5;
      }
      // density accept:  f(x) = (x/m)^a ((1 - x)/(1 - m))^b
      if (Math.log(W) <= a * Math.log(X / m) + b * Math.log((1.0 - X) / (1.0 - m))) {
        return X;
      }
    }
  }

  /** Returns the cumulative distribution function. */
  @Override
  public double cdf(double x) {
    return Probability.beta(alpha, beta, x);
  }

  private static double f(double x, double a, double b, double m) {
    return Math.exp(a * Math.log(x / m) + b * Math.log((1.0 - x) / (1.0 - m)));
  }

  /** Returns a random number from the distribution. */
  @Override
  public double nextDouble() {
    return nextDouble(alpha, beta);
  }

  /** Returns a beta distributed random number; bypasses the internal state. */
  public double nextDouble(double alpha, double beta) {
/******************************************************************
 *                                                                *
 * Beta Distribution - Stratified Rejection/Patchwork Rejection   *
 *                                                                *
 ******************************************************************
 * For parameters a < 1 , b < 1  and  a < 1 < b   or  b < 1 < a   *
 * the stratified rejection methods b00 and b01 of Sakasegawa are *
 * used. Both procedures employ suitable two-part power functions *
 * from which samples can be obtained by inversion.               *
 * If a > 1 , b > 1 (unimodal case) the patchwork rejection       *
 * method b1prs of Zechner/Stadlober is utilized:                 *
 * The area below the density function f(x) in its body is        *
 * rearranged by certain point reflections. Within a large center *
 * interval variates are sampled efficiently by rejection from    *
 * uniform hats. Rectangular immediate acceptance regions speed   *
 * up the generation. The remaining tails are covered by          *
 * exponential functions.                                         *
 * If (a-1)(b-1) = 0  sampling is done by inversion if either a   *
 * or b are not equal to one. If  a = b = 1  a uniform random     *
 * variate is delivered.                                          *
 *                                                                *
 ******************************************************************
 *                                                                *
 * FUNCTION :   - bsprc samples a random variate from the beta    *
 *                distribution with parameters  a > 0, b > 0.     *
 * REFERENCES : - H. Sakasegawa (1983): Stratified rejection and  *
 *                squeeze method for generating beta random       *
 *                numbers, Ann. Inst. Statist. Math. 35 B,        *
 *                291-302.                                        *
 *              - H. Zechner, E. Stadlober (1993): Generating     *
 *                beta variates via patchwork rejection,          *
 *                Computing 50, 1-18.                             *
 *                                                                *
 * SUBPROGRAMS: - drand(seed) ... (0,1)-Uniform generator with    *
 *                unsigned long integer *seed.                    *
 *              - b00(seed,a,b) ... Beta generator for a<1, b<1   *
 *              - b01(seed,a,b) ... Beta generator for a<1<b or   *
 *                                  b<1<a                         *
 *              - b1prs(seed,a,b) ... Beta generator for a>1, b>1 *
 *                with unsigned long integer *seed, double a, b.  *
 *                                                                *
 ******************************************************************/
    if (alpha > 1.0) {
      if (beta > 1.0) {
        return b1prs(alpha, beta, randomGenerator);
      }
      if (beta < 1.0) {
        return 1.0 - b01(beta, alpha, randomGenerator);
      }
      return Math.exp(Math.log(1 - randomGenerator.nextDouble()) / alpha);
    }

    if (alpha < 1.0) {
      if (beta > 1.0) {
        return b01(alpha, beta, randomGenerator);
      }
      if (beta < 1.0) {
        return b00(alpha, beta, randomGenerator);
      }
      return Math.exp(Math.log(randomGenerator.nextDouble()) / alpha);
    }

    return beta == 1.0 ? randomGenerator.nextDouble() : 1.0 - Math.exp(Math.log(randomGenerator.nextDouble()) / beta);
  }

  /** Returns the cumulative distribution function. */
  @Override
  public double pdf(double x) {
    if (x < 0 || x > 1) {
      return 0.0;
    }
    return Math.exp(pdfConst) * Math.pow(x, alpha - 1) * Math.pow(1 - x, beta - 1);
  }

  /** Sets the parameters. */
  public final void setState(double alpha, double beta) {
    this.alpha = alpha;
    this.beta = beta;
    this.pdfConst = Fun.logGamma(alpha + beta) - Fun.logGamma(alpha) - Fun.logGamma(beta);
  }

  /** Returns a String representation of the receiver. */
  public String toString() {
    return this.getClass().getName() + '(' + alpha + ',' + beta + ')';
  }

}
