/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.random;

import org.apache.mahout.math.jet.random.engine.RandomEngine;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class Distributions {

  private Distributions() {
  }

  /**
   * Returns the probability distribution function of the discrete geometric distribution. <p> <tt>p(k) = p *
   * (1-p)^k</tt> for <tt> k &gt;= 0</tt>. <p>
   *
   * @param k the argument to the probability distribution function.
   * @param p the parameter of the probability distribution function.
   */
  public static double geometricPdf(int k, double p) {
    if (k < 0) {
      throw new IllegalArgumentException();
    }
    return p * Math.pow(1 - p, k);
  }

  /**
   * Returns a random number from the Burr II, VII, VIII, X Distributions. <p> <b>Implementation:</b> Inversion method.
   * This is a port of <tt>burr1.c</tt> from the <A HREF="http://www.cis.tu-graz.ac.at/stat/stadl/random.html">C-RAND /
   * WIN-RAND</A> library. C-RAND's implementation, in turn, is based upon <p> L. Devroye (1986): Non-Uniform Random
   * Variate Generation, Springer Verlag, New York. <p>
   *
   * @param r  must be &gt; 0.
   * @param nr the number of the burr distribution (e.g. 2,7,8,10).
   */
  public static double nextBurr1(double r, int nr, RandomEngine randomGenerator) {
/******************************************************************
 *                                                                *
 *        Burr II, VII, VIII, X Distributions - Inversion         *
 *                                                                *
 ******************************************************************
 *                                                                *
 * FUNCTION :   - burr1 samples a random number from one of the   *
 *                Burr II, VII, VIII, X distributions with        *
 *                parameter  r > 0 , where the no. of the         *
 *                distribution is indicated by a pointer          *
 *                variable.                                       *
 * REFERENCE :  - L. Devroye (1986): Non-Uniform Random Variate   *
 *                Generation, Springer Verlag, New York.          *
 * SUBPROGRAM : - drand(seed) ... (0,1)-uniform generator with    *
 *                unsigned long integer *seed.                    *
 *                                                                *
 ******************************************************************/

    double y = Math.exp(Math.log(randomGenerator.raw()) / r);
    switch (nr) {
      // BURR II
      case 2:
        return (-Math.log(1 / y - 1));

      // BURR VII
      case 7:
        return (Math.log(2 * y / (2 - 2 * y)) / 2);

      // BURR VIII
      case 8:
        return (Math.log(Math.tan(y * Math.PI / 2.0)));

      // BURR X
      case 10:
        return (Math.sqrt(-Math.log(1 - y)));
    }
    return 0;
  }

  /**
   * Returns a random number from the Burr III, IV, V, VI, IX, XII distributions. <p> <b>Implementation:</b> Inversion
   * method. This is a port of <tt>burr2.c</tt> from the <A HREF="http://www.cis.tu-graz.ac.at/stat/stadl/random.html">C-RAND
   * / WIN-RAND</A> library. C-RAND's implementation, in turn, is based upon <p> L. Devroye (1986): Non-Uniform Random
   * Variate Generation, Springer Verlag, New York. <p>
   *
   * @param r  must be &gt; 0.
   * @param k  must be &gt; 0.
   * @param nr the number of the burr distribution (e.g. 3,4,5,6,9,12).
   */
  public static double nextBurr2(double r, double k, int nr, RandomEngine randomGenerator) {
/******************************************************************
 *                                                                *
 *      Burr III, IV, V, VI, IX, XII Distribution - Inversion     *
 *                                                                *
 ******************************************************************
 *                                                                *
 * FUNCTION :   - burr2 samples a random number from one of the   *
 *                Burr III, IV, V, VI, IX, XII distributions with *
 *                parameters r > 0 and k > 0, where the no. of    *
 *                the distribution is indicated by a pointer      *
 *                variable.                                       *
 * REFERENCE :  - L. Devroye (1986): Non-Uniform Random Variate   *
 *                Generation, Springer Verlag, New York.          *
 * SUBPROGRAM : - drand(seed) ... (0,1)-Uniform generator with    *
 *                unsigned long integer *seed.                    *
 *                                                                *
 ******************************************************************/
    double u = randomGenerator.raw();
    double y = Math.exp(-Math.log(u) / r) - 1.0;
    switch (nr) {
      case 3:               // BURR III
        return (Math.exp(-Math.log(y) / k));      // y^(-1/k)

      case 4:               // BURR IV
        y = Math.exp(k * Math.log(y)) + 1.0;         // y^k + 1
        y = k / y;
        return (y);

      case 5:               // BURR V
        y = Math.atan(-Math.log(y / k));           // arctan[log(y/k)]
        return (y);

      case 6:               // BURR VI
        y = -Math.log(y / k) / r;
        y = Math.log(y + Math.sqrt(y * y + 1.0));
        return (y);

      case 9:               // BURR IX
        y = 1.0 + 2.0 * u / (k * (1.0 - u));
        y = Math.exp(Math.log(y) / r) - 1.0;         // y^(1/r) -1
        return Math.log(y);

      case 12:               // BURR XII
        return Math.exp(Math.log(y) / k);        // y^(1/k)
    }
    return 0;
  }

  /**
   * Returns a cauchy distributed random number from the standard Cauchy distribution C(0,1). <A
   * HREF="http://www.cern.ch/RD11/rkb/AN16pp/node25.html#SECTION000250000000000000000"> math definition</A> and <A
   * HREF="http://www.statsoft.com/textbook/glosc.html#Cauchy Distribution"> animated definition</A>. <p> <tt>p(x) = 1/
   * (mean*pi * (1+(x/mean)^2))</tt>. <p> <b>Implementation:</b> This is a port of <tt>cin.c</tt> from the <A
   * HREF="http://www.cis.tu-graz.ac.at/stat/stadl/random.html">C-RAND / WIN-RAND</A> library. <p>
   */
  public static double nextCauchy(RandomEngine randomGenerator) {
    return Math.tan(Math.PI * randomGenerator.raw());
  }

  /** Returns an erlang distributed random number with the given variance and mean. */
  public static double nextErlang(double variance, double mean, RandomEngine randomGenerator) {
    int k = (int) ((mean * mean) / variance + 0.5);
    k = (k > 0) ? k : 1;
    double a = k / mean;

    double prod = 1.0;
    for (int i = 0; i < k; i++) {
      prod *= randomGenerator.raw();
    }
    return -Math.log(prod) / a;
  }

  /**
   * Returns a discrete geometric distributed random number; <A HREF="http://www.statsoft.com/textbook/glosf.html#Geometric
   * Distribution">Definition</A>. <p> <tt>p(k) = p * (1-p)^k</tt> for <tt> k &gt;= 0</tt>. <p> <b>Implementation:</b>
   * Inversion method. This is a port of <tt>geo.c</tt> from the <A HREF="http://www.cis.tu-graz.ac.at/stat/stadl/random.html">C-RAND
   * / WIN-RAND</A> library.
   *
   * @param p must satisfy <tt>0 &lt; p &lt; 1</tt>. <p>
   */
  public static int nextGeometric(double p, RandomEngine randomGenerator) {
/******************************************************************
 *                                                                *
 *              Geometric Distribution - Inversion                *
 *                                                                *
 ******************************************************************
 *                                                                *
 * On generating random numbers of a discrete distribution by     *
 * Inversion normally sequential search is necessary, but in the  *
 * case of the Geometric distribution a direct transformation is  *
 * possible because of the special parallel to the continuous     *
 * Exponential distribution Exp(t):                               *
 *    X - Exp(t): G(x)=1-exp(-tx)                                 *
 *        Geo(p): pk=G(k+1)-G(k)=exp(-tk)*(1-exp(-t))             *
 *                p=1-exp(-t)                                     *
 * A random number of the Geometric distribution Geo(p) is        *
 * obtained by k=(long int)x, where x is from Exp(t) with         *
 * parameter t=-log(1-p).                                         *
 *                                                                *
 ******************************************************************
 *                                                                *
 * FUNCTION:    - geo samples a random number from the Geometric  *
 *                distribution with parameter 0<p<1.              *
 * SUBPROGRAMS: - drand(seed) ... (0,1)-Uniform generator with    *
 *                unsigned long integer *seed.                    *
 *                                                                *
 ******************************************************************/
    double u = randomGenerator.raw();
    return (int) (Math.log(u) / Math.log(1.0 - p));
  }

  /**
   * Returns a lambda distributed random number with parameters l3 and l4. <p> <b>Implementation:</b> Inversion method.
   * This is a port of <tt>lamin.c</tt> from the <A HREF="http://www.cis.tu-graz.ac.at/stat/stadl/random.html">C-RAND /
   * WIN-RAND</A> library. C-RAND's implementation, in turn, is based upon <p> J.S. Ramberg, B:W. Schmeiser (1974): An
   * approximate method for generating asymmetric variables, Communications ACM 17, 78-82. <p>
   */
  public static double nextLambda(double l3, double l4, RandomEngine randomGenerator) {
    double l_sign;
    if ((l3 < 0) || (l4 < 0)) {
      l_sign = -1.0;                          // sign(l)
    } else {
      l_sign = 1.0;
    }

    double u = randomGenerator.raw();                           // U(0/1)
    return l_sign * (Math.exp(Math.log(u) * l3) - Math.exp(Math.log(1.0 - u) * l4));
  }

  /**
   * Returns a Laplace (Double Exponential) distributed random number from the standard Laplace distribution L(0,1). <p>
   * <b>Implementation:</b> Inversion method. This is a port of <tt>lapin.c</tt> from the <A
   * HREF="http://www.cis.tu-graz.ac.at/stat/stadl/random.html">C-RAND / WIN-RAND</A> library. <p>
   */
  public static double nextLaplace(RandomEngine randomGenerator) {
    double u = randomGenerator.raw();
    u = u + u - 1.0;
    if (u > 0) {
      return -Math.log(1.0 - u);
    } else {
      return Math.log(1.0 + u);
    }
  }

  /**
   * Returns a random number from the standard Logistic distribution Log(0,1). <p> <b>Implementation:</b> Inversion
   * method. This is a port of <tt>login.c</tt> from the <A HREF="http://www.cis.tu-graz.ac.at/stat/stadl/random.html">C-RAND
   * / WIN-RAND</A> library.
   */
  public static double nextLogistic(RandomEngine randomGenerator) {
    double u = randomGenerator.raw();
    return (-Math.log(1.0 / u - 1.0));
  }

  /**
   * Returns a power-law distributed random number with the given exponent and lower cutoff.
   *
   * @param alpha the exponent
   * @param cut   the lower cutoff
   */
  public static double nextPowLaw(double alpha, double cut, RandomEngine randomGenerator) {
    return cut * Math.pow(randomGenerator.raw(), 1.0 / (alpha + 1.0));
  }

  /**
   * Returns a random number from the standard Triangular distribution in (-1,1). <p> <b>Implementation:</b> Inversion
   * method. This is a port of <tt>tra.c</tt> from the <A HREF="http://www.cis.tu-graz.ac.at/stat/stadl/random.html">C-RAND
   * / WIN-RAND</A> library. <p>
   */
  public static double nextTriangular(RandomEngine randomGenerator) {
/******************************************************************
 *                                                                *
 *     Triangular Distribution - Inversion: x = +-(1-sqrt(u))     *
 *                                                                *
 ******************************************************************
 *                                                                *
 * FUNCTION :   - tra samples a random number from the            *
 *                standard Triangular distribution in (-1,1)      *
 * SUBPROGRAM : - drand(seed) ... (0,1)-Uniform generator with    *
 *                unsigned long integer *seed.                    *
 *                                                                *
 ******************************************************************/

    double u = randomGenerator.raw();
    if (u <= 0.5) {
      return (Math.sqrt(2.0 * u) - 1.0);                      /* -1 <= x <= 0 */
    } else {
      return (1.0 - Math.sqrt(2.0 * (1.0 - u)));
    }                 /*  0 <= x <= 1 */
  }

  /**
   * Returns a weibull distributed random number. Polar method. See Simulation, Modelling & Analysis by Law & Kelton,
   * pp259
   */
  public static double nextWeibull(double alpha, double beta, RandomEngine randomGenerator) {
    // Polar method.
    // See Simulation, Modelling & Analysis by Law & Kelton, pp259
    return Math.pow(beta * (-Math.log(1.0 - randomGenerator.raw())), 1.0 / alpha);
  }

  /**
   * Returns a zipfian distributed random number with the given skew. <p> Algorithm from page 551 of: Devroye, Luc
   * (1986) `Non-uniform random variate generation', Springer-Verlag: Berlin.   ISBN 3-540-96305-7 (also 0-387-96305-7)
   *
   * @param z the skew of the distribution (must be &gt;1.0).
   */
  public static int nextZipfInt(double z, RandomEngine randomGenerator) {
    /* Algorithm from page 551 of:
    * Devroye, Luc (1986) `Non-uniform random variate generation',
    * Springer-Verlag: Berlin.   ISBN 3-540-96305-7 (also 0-387-96305-7)
    */
    double b = Math.pow(2.0, z - 1.0);
    double constant = -1.0 / (z - 1.0);

    int result;
    while (true) {
      double u = randomGenerator.raw();
      double v = randomGenerator.raw();
      result = (int) (Math.floor(Math.pow(u, constant)));
      double t = Math.pow(1.0 + 1.0 / result, z - 1.0);
      if (v * result * (t - 1.0) / (b - 1.0) <= t / b) {
        break;
      }
    }
    return result;
  }
}
