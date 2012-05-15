/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.stat;

import org.apache.mahout.math.jet.math.Constants;
import org.apache.mahout.math.jet.math.Polynomial;

/** Partially deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
public final class Gamma {

  private static final double MAXSTIR = 143.01608;

  private Gamma() {
  }

  /**
   * Returns the beta function of the arguments.
   * <pre>
   *                   -     -
   *                  | (a) | (b)
   * beta( a, b )  =  -----------.
   *                     -
   *                    | (a+b)
   * </pre>
   * @param alpha
   * @param beta
   * @return The beta function for given values of alpha and beta.
   */
  public static double beta(double alpha, double beta) {
    double y;
    if (alpha < 40 && beta < 40) {
      y = gamma(alpha + beta);
      if (y == 0.0) {
        return 1.0;
      }

      if (alpha > beta) {
        y = gamma(alpha) / y;
        y *= gamma(beta);
      } else {
        y = gamma(beta) / y;
        y *= gamma(alpha);
      }
    } else {
      y = Math.exp(logGamma(alpha) + logGamma(beta) - logGamma(alpha + beta));
    }

    return y;
  }

  /** Returns the Gamma function of the argument. */
  public static double gamma(double x) {

    double[] pCoefficient = {
      1.60119522476751861407E-4,
      1.19135147006586384913E-3,
      1.04213797561761569935E-2,
      4.76367800457137231464E-2,
      2.07448227648435975150E-1,
      4.94214826801497100753E-1,
      9.99999999999999996796E-1
    };
    double[] qCoefficient = {
      -2.31581873324120129819E-5,
      5.39605580493303397842E-4,
      -4.45641913851797240494E-3,
      1.18139785222060435552E-2,
      3.58236398605498653373E-2,
      -2.34591795718243348568E-1,
      7.14304917030273074085E-2,
      1.00000000000000000320E0
    };
//double MAXGAM = 171.624376956302725;
//double LOGPI  = 1.14472988584940017414;

    double p;
    double z;

    double q = Math.abs(x);

    if (q > 33.0) {
      if (x < 0.0) {
        p = Math.floor(q);
        if (p == q) {
          throw new ArithmeticException("gamma: overflow");
        }
        //int i = (int) p;
        z = q - p;
        if (z > 0.5) {
          p += 1.0;
          z = q - p;
        }
        z = q * Math.sin(Math.PI * z);
        if (z == 0.0) {
          throw new ArithmeticException("gamma: overflow");
        }
        z = Math.abs(z);
        z = Math.PI / (z * stirlingFormula(q));

        return -z;
      } else {
        return stirlingFormula(x);
      }
    }

    z = 1.0;
    while (x >= 3.0) {
      x -= 1.0;
      z *= x;
    }

    while (x < 0.0) {
      if (x == 0.0) {
        throw new ArithmeticException("gamma: singular");
      }
      if (x > -1.0e-9) {
        return z / ((1.0 + 0.5772156649015329 * x) * x);
      }
      z /= x;
      x += 1.0;
    }

    while (x < 2.0) {
      if (x == 0.0) {
        throw new ArithmeticException("gamma: singular");
      }
      if (x < 1.0e-9) {
        return z / ((1.0 + 0.5772156649015329 * x) * x);
      }
      z /= x;
      x += 1.0;
    }

    if ((x == 2.0) || (x == 3.0)) {
      return z;
    }

    x -= 2.0;
    p = Polynomial.polevl(x, pCoefficient, 6);
    q = Polynomial.polevl(x, qCoefficient, 7);
    return z * p / q;

  }

  /**
   * Returns the regularized Incomplete Beta Function evaluated from zero to <tt>xx</tt>; formerly named <tt>ibeta</tt>.
   *
   * See http://en.wikipedia.org/wiki/Incomplete_beta_function#Incomplete_beta_function
   *
   * @param alpha the alpha parameter of the beta distribution.
   * @param beta the beta parameter of the beta distribution.
   * @param xx the integration end point.
   */
  public static double incompleteBeta(double alpha, double beta, double xx) {

    if (alpha <= 0.0) {
      throw new ArithmeticException("incompleteBeta: Domain error! alpha must be > 0, but was " + alpha);
    }

    if (beta <= 0.0) {
      throw new ArithmeticException("incompleteBeta: Domain error! beta must be > 0, but was " + beta);
    }

    if (xx <= 0.0) {
      return 0.0;
    }

    if (xx >= 1.0) {
      return 1.0;
    }

    double t;
    if ((beta * xx) <= 1.0 && xx <= 0.95) {
      t = powerSeries(alpha, beta, xx);
      return t;
    }

    double w = 1.0 - xx;

    /* Reverse a and b if x is greater than the mean. */
    double xc;
    double x;
    double b;
    double a;
    boolean flag = false;
    if (xx > (alpha / (alpha + beta))) {
      flag = true;
      a = beta;
      b = alpha;
      xc = xx;
      x = w;
    } else {
      a = alpha;
      b = beta;
      xc = w;
      x = xx;
    }

    if (flag && (b * x) <= 1.0 && x <= 0.95) {
      t = powerSeries(a, b, x);
      t = t <= Constants.MACHEP ? 1.0 - Constants.MACHEP : 1.0 - t;
      return t;
    }

    /* Choose expansion for better convergence. */
    double y = x * (a + b - 2.0) - (a - 1.0);
    w = y < 0.0 ? incompleteBetaFraction1(a, b, x) : incompleteBetaFraction2(a, b, x) / xc;

    /* Multiply w by the factor
       a      b   _             _     _
      x  (1-x)   | (a+b) / ( a | (a) | (b) ) .   */

    y = a * Math.log(x);
    t = b * Math.log(xc);
    if ((a + b) < Constants.MAXGAM && Math.abs(y) < Constants.MAXLOG && Math.abs(t) < Constants.MAXLOG) {
      t = Math.pow(xc, b);
      t *= Math.pow(x, a);
      t /= a;
      t *= w;
      t *= gamma(a + b) / (gamma(a) * gamma(b));
      if (flag) {
        t = t <= Constants.MACHEP ? 1.0 - Constants.MACHEP : 1.0 - t;
      }
      return t;
    }
    /* Resort to logarithms.  */
    y += t + logGamma(a + b) - logGamma(a) - logGamma(b);
    y += Math.log(w / a);
    t = y < Constants.MINLOG ? 0.0 : Math.exp(y);

    if (flag) {
      t = t <= Constants.MACHEP ? 1.0 - Constants.MACHEP : 1.0 - t;
    }
    return t;
  }

  /** Continued fraction expansion #1 for incomplete beta integral; formerly named <tt>incbcf</tt>. */
  static double incompleteBetaFraction1(double a, double b, double x) {

    double k1 = a;
    double k2 = a + b;
    double k3 = a;
    double k4 = a + 1.0;
    double k5 = 1.0;
    double k6 = b - 1.0;
    double k7 = k4;
    double k8 = a + 2.0;

    double pkm2 = 0.0;
    double qkm2 = 1.0;
    double pkm1 = 1.0;
    double qkm1 = 1.0;
    double ans = 1.0;
    double r = 1.0;
    int n = 0;
    double thresh = 3.0 * Constants.MACHEP;
    do {
      double xk = -(x * k1 * k2) / (k3 * k4);
      double pk = pkm1 + pkm2 * xk;
      double qk = qkm1 + qkm2 * xk;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;

      xk = (x * k5 * k6) / (k7 * k8);
      pk = pkm1 + pkm2 * xk;
      qk = qkm1 + qkm2 * xk;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;

      if (qk != 0) {
        r = pk / qk;
      }
      double t;
      if (r != 0) {
        t = Math.abs((ans - r) / r);
        ans = r;
      } else {
        t = 1.0;
      }

      if (t < thresh) {
        return ans;
      }

      k1 += 1.0;
      k2 += 1.0;
      k3 += 2.0;
      k4 += 2.0;
      k5 += 1.0;
      k6 -= 1.0;
      k7 += 2.0;
      k8 += 2.0;

      if ((Math.abs(qk) + Math.abs(pk)) > Constants.BIG) {
        pkm2 *= Constants.BIG_INVERSE;
        pkm1 *= Constants.BIG_INVERSE;
        qkm2 *= Constants.BIG_INVERSE;
        qkm1 *= Constants.BIG_INVERSE;
      }
      if ((Math.abs(qk) < Constants.BIG_INVERSE) || (Math.abs(pk) < Constants.BIG_INVERSE)) {
        pkm2 *= Constants.BIG;
        pkm1 *= Constants.BIG;
        qkm2 *= Constants.BIG;
        qkm1 *= Constants.BIG;
      }
    } while (++n < 300);

    return ans;
  }

  /** Continued fraction expansion #2 for incomplete beta integral; formerly named <tt>incbd</tt>. */
  static double incompleteBetaFraction2(double a, double b, double x) {

    double k1 = a;
    double k2 = b - 1.0;
    double k3 = a;
    double k4 = a + 1.0;
    double k5 = 1.0;
    double k6 = a + b;
    double k7 = a + 1.0;
    double k8 = a + 2.0;

    double pkm2 = 0.0;
    double qkm2 = 1.0;
    double pkm1 = 1.0;
    double qkm1 = 1.0;
    double z = x / (1.0 - x);
    double ans = 1.0;
    double r = 1.0;
    int n = 0;
    double thresh = 3.0 * Constants.MACHEP;
    do {
      double xk = -(z * k1 * k2) / (k3 * k4);
      double pk = pkm1 + pkm2 * xk;
      double qk = qkm1 + qkm2 * xk;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;

      xk = (z * k5 * k6) / (k7 * k8);
      pk = pkm1 + pkm2 * xk;
      qk = qkm1 + qkm2 * xk;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;

      if (qk != 0) {
        r = pk / qk;
      }
      double t;
      if (r != 0) {
        t = Math.abs((ans - r) / r);
        ans = r;
      } else {
        t = 1.0;
      }

      if (t < thresh) {
        return ans;
      }

      k1 += 1.0;
      k2 -= 1.0;
      k3 += 2.0;
      k4 += 2.0;
      k5 += 1.0;
      k6 += 1.0;
      k7 += 2.0;
      k8 += 2.0;

      if ((Math.abs(qk) + Math.abs(pk)) > Constants.BIG) {
        pkm2 *= Constants.BIG_INVERSE;
        pkm1 *= Constants.BIG_INVERSE;
        qkm2 *= Constants.BIG_INVERSE;
        qkm1 *= Constants.BIG_INVERSE;
      }
      if ((Math.abs(qk) < Constants.BIG_INVERSE) || (Math.abs(pk) < Constants.BIG_INVERSE)) {
        pkm2 *= Constants.BIG;
        pkm1 *= Constants.BIG;
        qkm2 *= Constants.BIG;
        qkm1 *= Constants.BIG;
      }
    } while (++n < 300);

    return ans;
  }

  /**
   * Returns the Incomplete Gamma function; formerly named <tt>igamma</tt>.
   *
   * @param alpha the shape parameter of the gamma distribution.
   * @param x the integration end point.
   * @return The value of the unnormalized incomplete gamma function.
   */
  public static double incompleteGamma(double alpha, double x) {
    if (x <= 0 || alpha <= 0) {
      return 0.0;
    }

    if (x > 1.0 && x > alpha) {
      return 1.0 - incompleteGammaComplement(alpha, x);
    }

    /* Compute  x**a * exp(-x) / gamma(a)  */
    double ax = alpha * Math.log(x) - x - logGamma(alpha);
    if (ax < -Constants.MAXLOG) {
      return 0.0;
    }

    ax = Math.exp(ax);

    /* power series */
    double r = alpha;
    double c = 1.0;
    double ans = 1.0;

    do {
      r += 1.0;
      c *= x / r;
      ans += c;
    }
    while (c / ans > Constants.MACHEP);

    return ans * ax / alpha;

  }

  /**
   * Returns the Complemented Incomplete Gamma function; formerly named <tt>igamc</tt>.
   *
   * @param alpha the shape parameter of the gamma distribution.
   * @param x the integration start point.
   */
  public static double incompleteGammaComplement(double alpha, double x) {

    if (x <= 0 || alpha <= 0) {
      return 1.0;
    }

    if (x < 1.0 || x < alpha) {
      return 1.0 - incompleteGamma(alpha, x);
    }

    double ax = alpha * Math.log(x) - x - logGamma(alpha);
    if (ax < -Constants.MAXLOG) {
      return 0.0;
    }

    ax = Math.exp(ax);

    /* continued fraction */
    double y = 1.0 - alpha;
    double z = x + y + 1.0;
    double c = 0.0;
    double pkm2 = 1.0;
    double qkm2 = x;
    double pkm1 = x + 1.0;
    double qkm1 = z * x;
    double ans = pkm1 / qkm1;

    double t;
    do {
      c += 1.0;
      y += 1.0;
      z += 2.0;
      double yc = y * c;
      double pk = pkm1 * z - pkm2 * yc;
      double qk = qkm1 * z - qkm2 * yc;
      if (qk != 0) {
        double r = pk / qk;
        t = Math.abs((ans - r) / r);
        ans = r;
      } else {
        t = 1.0;
      }

      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;
      if (Math.abs(pk) > Constants.BIG) {
        pkm2 *= Constants.BIG_INVERSE;
        pkm1 *= Constants.BIG_INVERSE;
        qkm2 *= Constants.BIG_INVERSE;
        qkm1 *= Constants.BIG_INVERSE;
      }
    } while (t > Constants.MACHEP);

    return ans * ax;
  }

  /** Returns the natural logarithm of the gamma function; formerly named <tt>lgamma</tt>. */
  public static double logGamma(double x) {
    double p;
    double q;
    double z;

    double[] aCoefficient = {
      8.11614167470508450300E-4,
      -5.95061904284301438324E-4,
      7.93650340457716943945E-4,
      -2.77777777730099687205E-3,
      8.33333333333331927722E-2
    };
    double[] bCoefficient = {
      -1.37825152569120859100E3,
      -3.88016315134637840924E4,
      -3.31612992738871184744E5,
      -1.16237097492762307383E6,
      -1.72173700820839662146E6,
      -8.53555664245765465627E5
    };
    double[] cCoefficient = {
      /* 1.00000000000000000000E0, */
      -3.51815701436523470549E2,
      -1.70642106651881159223E4,
      -2.20528590553854454839E5,
      -1.13933444367982507207E6,
      -2.53252307177582951285E6,
      -2.01889141433532773231E6
    };

    if (x < -34.0) {
      q = -x;
      double w = logGamma(q);
      p = Math.floor(q);
      if (p == q) {
        throw new ArithmeticException("lgam: Overflow");
      }
      z = q - p;
      if (z > 0.5) {
        p += 1.0;
        z = p - q;
      }
      z = q * Math.sin(Math.PI * z);
      if (z == 0.0) {
        throw new
            ArithmeticException("lgamma: Overflow");
      }
      z = Constants.LOGPI - Math.log(z) - w;
      return z;
    }

    if (x < 13.0) {
      z = 1.0;
      while (x >= 3.0) {
        x -= 1.0;
        z *= x;
      }
      while (x < 2.0) {
        if (x == 0.0) {
          throw new ArithmeticException("lgamma: Overflow");
        }
        z /= x;
        x += 1.0;
      }
      if (z < 0.0) {
        z = -z;
      }
      if (x == 2.0) {
        return Math.log(z);
      }
      x -= 2.0;
      p = x * Polynomial.polevl(x, bCoefficient, 5) / Polynomial.p1evl(x, cCoefficient, 6);
      return Math.log(z) + p;
    }

    if (x > 2.556348e305) {
      throw new ArithmeticException("lgamma: Overflow");
    }

    q = (x - 0.5) * Math.log(x) - x + 0.91893853320467274178;
    //if ( x > 1.0e8 ) return( q );
    if (x > 1.0e8) {
      return q;
    }

    p = 1.0 / (x * x);
    if (x >= 1000.0) {
      q += ((7.9365079365079365079365e-4 * p
          - 2.7777777777777777777778e-3) * p
          + 0.0833333333333333333333) / x;
    } else {
      q += Polynomial.polevl(p, aCoefficient, 4) / x;
    }
    return q;
  }

  /**
   * Power series for incomplete beta integral; formerly named <tt>pseries</tt>. Use when b*x is small and x not too
   * close to 1.
   */
  private static double powerSeries(double a, double b, double x) {

    double ai = 1.0 / a;
    double u = (1.0 - b) * x;
    double v = u / (a + 1.0);
    double t1 = v;
    double t = u;
    double n = 2.0;
    double s = 0.0;
    double z = Constants.MACHEP * ai;
    while (Math.abs(v) > z) {
      u = (n - b) * x / n;
      t *= u;
      v = t / (a + n);
      s += v;
      n += 1.0;
    }
    s += t1;
    s += ai;

    u = a * Math.log(x);
    if ((a + b) < Constants.MAXGAM && Math.abs(u) < Constants.MAXLOG) {
      t = gamma(a + b) / (gamma(a) * gamma(b));
      s *= t * Math.pow(x, a);
    } else {
      t = logGamma(a + b) - logGamma(a) - logGamma(b) + u + Math.log(s);
      s = t < Constants.MINLOG ? 0.0 : Math.exp(t);
    }
    return s;
  }

  /**
   * Returns the Gamma function computed by Stirling's formula; formerly named <tt>stirf</tt>. The polynomial STIR is
   * valid for 33 <= x <= 172.
   */
  static double stirlingFormula(double x) {
    double[] coefficients = {
      7.87311395793093628397E-4,
      -2.29549961613378126380E-4,
      -2.68132617805781232825E-3,
      3.47222221605458667310E-3,
      8.33333333333482257126E-2,
    };

    double w = 1.0 / x;
    double y = Math.exp(x);

    w = 1.0 + w * Polynomial.polevl(w, coefficients, 4);

    if (x > MAXSTIR) {
      /* Avoid overflow in Math.pow() */
      double v = Math.pow(x, 0.5 * x - 0.25);
      y = v * (v / y);
    } else {
      y = Math.pow(x, x - 0.5) / y;
    }
    y = Constants.SQTPI * y * w;
    return y;
  }
}
