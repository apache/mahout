/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.math;

/**
 * Bessel and Airy functions.
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class Bessel extends Constants {
  /****************************************
   *    COEFFICIENTS FOR METHODS i0, i0e  *
   ****************************************/

  /**
   * Chebyshev coefficients for exp(-x) I0(x) in the interval [0,8].
   *
   * lim(x->0){ exp(-x) I0(x) } = 1.
   */
  static final double[] aSubI0 = {
    -4.41534164647933937950E-18,
    3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
    1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
    7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
    2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
    9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
    2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
    6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
    1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
    1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
    1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
    1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
    1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
    4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
    1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
    6.76795274409476084995E-1
  };


  /**
   * Chebyshev coefficients for exp(-x) sqrt(x) I0(x) in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I0(x) } = 1/sqrt(2pi).
   */
  static final double[] bSubI0 = {
    -7.23318048787475395456E-18,
    -4.83050448594418207126E-18,
    4.46562142029675999901E-17,
    3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
    1.77256013305652638360E-15,
    3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
    1.54008621752140982691E-14,
    3.85277838274214270114E-13,
    7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
    1.18891471078464383424E-11,
    4.94060238822496958910E-10,
    3.39623202570838634515E-9,
    2.26666899049817806459E-8,
    2.04891858946906374183E-7,
    2.89137052083475648297E-6,
    6.88975834691682398426E-5,
    3.36911647825569408990E-3,
    8.04490411014108831608E-1
  };


  /****************************************
   *    COEFFICIENTS FOR METHODS i1, i1e  *
   ****************************************/
  /**
   * Chebyshev coefficients for exp(-x) I1(x) / x in the interval [0,8].
   *
   * lim(x->0){ exp(-x) I1(x) / x } = 1/2.
   */
  static final double[] aSubI1 = {
    2.77791411276104639959E-18,
    -2.11142121435816608115E-17,
    1.55363195773620046921E-16,
    -1.10559694773538630805E-15,
    7.60068429473540693410E-15,
    -5.04218550472791168711E-14,
    3.22379336594557470981E-13,
    -1.98397439776494371520E-12,
    1.17361862988909016308E-11,
    -6.66348972350202774223E-11,
    3.62559028155211703701E-10,
    -1.88724975172282928790E-9,
    9.38153738649577178388E-9,
    -4.44505912879632808065E-8,
    2.00329475355213526229E-7,
    -8.56872026469545474066E-7,
    3.47025130813767847674E-6,
    -1.32731636560394358279E-5,
    4.78156510755005422638E-5,
    -1.61760815825896745588E-4,
    5.12285956168575772895E-4,
    -1.51357245063125314899E-3,
    4.15642294431288815669E-3,
    -1.05640848946261981558E-2,
    2.47264490306265168283E-2,
    -5.29459812080949914269E-2,
    1.02643658689847095384E-1,
    -1.76416518357834055153E-1,
    2.52587186443633654823E-1
  };

  /*
   * Chebyshev coefficients for exp(-x) sqrt(x) I1(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I1(x) } = 1/sqrt(2pi).
   */
  static final double[] bSubI1 = {
    7.51729631084210481353E-18,
    4.41434832307170791151E-18,
    -4.65030536848935832153E-17,
    -3.20952592199342395980E-17,
    2.96262899764595013876E-16,
    3.30820231092092828324E-16,
    -1.88035477551078244854E-15,
    -3.81440307243700780478E-15,
    1.04202769841288027642E-14,
    4.27244001671195135429E-14,
    -2.10154184277266431302E-14,
    -4.08355111109219731823E-13,
    -7.19855177624590851209E-13,
    2.03562854414708950722E-12,
    1.41258074366137813316E-11,
    3.25260358301548823856E-11,
    -1.89749581235054123450E-11,
    -5.58974346219658380687E-10,
    -3.83538038596423702205E-9,
    -2.63146884688951950684E-8,
    -2.51223623787020892529E-7,
    -3.88256480887769039346E-6,
    -1.10588938762623716291E-4,
    -9.76109749136146840777E-3,
    7.78576235018280120474E-1
  };


  /** ********************************** COEFFICIENTS FOR METHODS k0, k0e  * ************************************** */
  /* Chebyshev coefficients for K0(x) + log(x/2) I0(x)
   * in the interval [0,2].  The odd order coefficients are all
   * zero; only the even order coefficients are listed.
   * 
   * lim(x->0){ K0(x) + log(x/2) I0(x) } = -EUL.
   */
  static final double[] aSubK0 = {
    1.37446543561352307156E-16,
    4.25981614279661018399E-14,
    1.03496952576338420167E-11,
    1.90451637722020886025E-9,
    2.53479107902614945675E-7,
    2.28621210311945178607E-5,
    1.26461541144692592338E-3,
    3.59799365153615016266E-2,
    3.44289899924628486886E-1,
    -5.35327393233902768720E-1
  };

  /* Chebyshev coefficients for exp(x) sqrt(x) K0(x)
   * in the inverted interval [2,infinity].
   * 
   * lim(x->inf){ exp(x) sqrt(x) K0(x) } = sqrt(pi/2).
   */
  static final double[] bSubK0 = {
    5.30043377268626276149E-18,
    -1.64758043015242134646E-17,
    5.21039150503902756861E-17,
    -1.67823109680541210385E-16,
    5.51205597852431940784E-16,
    -1.84859337734377901440E-15,
    6.34007647740507060557E-15,
    -2.22751332699166985548E-14,
    8.03289077536357521100E-14,
    -2.98009692317273043925E-13,
    1.14034058820847496303E-12,
    -4.51459788337394416547E-12,
    1.85594911495471785253E-11,
    -7.95748924447710747776E-11,
    3.57739728140030116597E-10,
    -1.69753450938905987466E-9,
    8.57403401741422608519E-9,
    -4.66048989768794782956E-8,
    2.76681363944501510342E-7,
    -1.83175552271911948767E-6,
    1.39498137188764993662E-5,
    -1.28495495816278026384E-4,
    1.56988388573005337491E-3,
    -3.14481013119645005427E-2,
    2.44030308206595545468E0
  };


  /** ********************************** COEFFICIENTS FOR METHODS k1, k1e  * ************************************** */
  /* Chebyshev coefficients for x(K1(x) - log(x/2) I1(x))
   * in the interval [0,2].
   * 
   * lim(x->0){ x(K1(x) - log(x/2) I1(x)) } = 1.
   */
  static final double[] aSubK1 = {
    -7.02386347938628759343E-18,
    -2.42744985051936593393E-15,
    -6.66690169419932900609E-13,
    -1.41148839263352776110E-10,
    -2.21338763073472585583E-8,
    -2.43340614156596823496E-6,
    -1.73028895751305206302E-4,
    -6.97572385963986435018E-3,
    -1.22611180822657148235E-1,
    -3.53155960776544875667E-1,
    1.52530022733894777053E0
  };

  /* Chebyshev coefficients for exp(x) sqrt(x) K1(x)
   * in the interval [2,infinity].
   *
   * lim(x->inf){ exp(x) sqrt(x) K1(x) } = sqrt(pi/2).
   */
  static final double[] bSubK1 = {
    -5.75674448366501715755E-18,
    1.79405087314755922667E-17,
    -5.68946255844285935196E-17,
    1.83809354436663880070E-16,
    -6.05704724837331885336E-16,
    2.03870316562433424052E-15,
    -7.01983709041831346144E-15,
    2.47715442448130437068E-14,
    -8.97670518232499435011E-14,
    3.34841966607842919884E-13,
    -1.28917396095102890680E-12,
    5.13963967348173025100E-12,
    -2.12996783842756842877E-11,
    9.21831518760500529508E-11,
    -4.19035475934189648750E-10,
    2.01504975519703286596E-9,
    -1.03457624656780970260E-8,
    5.74108412545004946722E-8,
    -3.50196060308781257119E-7,
    2.40648494783721712015E-6,
    -1.93619797416608296024E-5,
    1.95215518471351631108E-4,
    -2.85781685962277938680E-3,
    1.03923736576817238437E-1,
    2.72062619048444266945E0
  };
  private static final double EULER_GAMMA = 5.772156649015328606065e-1;

  /** Makes this class non instantiable, but still let's others inherit from it. */
  protected Bessel() {
  }

  /**
   * Returns the modified Bessel function of order 0 of the argument. <p> The function is defined as <tt>i0(x) = j0( ix
   * )</tt>. <p> The range is partitioned into the two intervals [0,8] and (8, infinity).  Chebyshev polynomial
   * expansions are employed in each interval.
   *
   * @param x the value to compute the bessel function of.
   */
  public static double i0(double x) {
    if (x < 0) {
      x = -x;
    }
    if (x <= 8.0) {
      return Math.exp(x) * Arithmetic.chbevl((x / 2.0) - 2.0, aSubI0, 30);
    }

    return Math.exp(x) * Arithmetic.chbevl(32.0 / x - 2.0, bSubI0, 25) / Math.sqrt(x);
  }

  /**
   * Returns the exponentially scaled modified Bessel function of order 0 of the argument. <p> The function is defined
   * as <tt>i0e(x) = exp(-|x|) j0( ix )</tt>.
   *
   * @param x the value to compute the bessel function of.
   */
  public static double i0e(double x) {

    if (x < 0) {
      x = -x;
    }
    if (x <= 8.0) {
      return Arithmetic.chbevl((x / 2.0) - 2.0, aSubI0, 30);
    }

    return Arithmetic.chbevl(32.0 / x - 2.0, bSubI0, 25) / Math.sqrt(x);
  }

  /**
   * Returns the modified Bessel function of order 1 of the argument. <p> The function is defined as <tt>i1(x) = -i j1(
   * ix )</tt>. <p> The range is partitioned into the two intervals [0,8] and (8, infinity).  Chebyshev polynomial
   * expansions are employed in each interval.
   *
   * @param x the value to compute the bessel function of.
   */
  public static double i1(double x) {

    double z = Math.abs(x);
    if (z <= 8.0) {
      z = Arithmetic.chbevl((z / 2.0) - 2.0, aSubI1, 29) * z * Math.exp(z);
    } else {
      z = Math.exp(z) * Arithmetic.chbevl(32.0 / z - 2.0, bSubI1, 25) / Math.sqrt(z);
    }
    if (x < 0.0) {
      z = -z;
    }
    return z;
  }

  /**
   * Returns the exponentially scaled modified Bessel function of order 1 of the argument. <p> The function is defined
   * as <tt>i1(x) = -i exp(-|x|) j1( ix )</tt>.
   *
   * @param x the value to compute the bessel function of.
   */
  public static double i1e(double x) {

    double z = Math.abs(x);
    if (z <= 8.0) {
      z = Arithmetic.chbevl((z / 2.0) - 2.0, aSubI1, 29) * z;
    } else {
      z = Arithmetic.chbevl(32.0 / z - 2.0, bSubI1, 25) / Math.sqrt(z);
    }
    if (x < 0.0) {
      z = -z;
    }
    return z;
  }

  /**
   * Returns the Bessel function of the first kind of order 0 of the argument.
   *
   * @param x the value to compute the bessel function of.
   */
  public static double j0(double x) {
    double ax;

    if ((ax = Math.abs(x)) < 8.0) {
      double y = x * x;
      double ans1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7
          + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
      double ans2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718
          + y * (59272.64853 + y * (267.8532712 + y))));

      return ans1 / ans2;

    } else {
      double z = 8.0 / ax;
      double y = z * z;
      double xx = ax - 0.785398164;
      double ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4
          + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
      double ans2 = -0.1562499995e-1 + y * (0.1430488765e-3
          + y * (-0.6911147651e-5 + y * (0.7621095161e-6
          - y * 0.934935152e-7)));

      return Math.sqrt(0.636619772 / ax) * (Math.cos(xx) * ans1 - z * Math.sin(xx) * ans2);
    }
  }

  /**
   * Returns the Bessel function of the first kind of order 1 of the argument.
   *
   * @param x the value to compute the bessel function of.
   */
  public static double j1(double x) {
    double ax;
    double y;
    double ans1;
    double ans2;

    if ((ax = Math.abs(x)) < 8.0) {
      y = x * x;
      ans1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1
          + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
      ans2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74
          + y * (99447.43394 + y * (376.9991397 + y))));
      return ans1 / ans2;
    } else {
      double z = 8.0 / ax;
      double xx = ax - 2.356194491;
      y = z * z;

      ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4
          + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
      ans2 = 0.04687499995 + y * (-0.2002690873e-3
          + y * (0.8449199096e-5 + y * (-0.88228987e-6
          + y * 0.105787412e-6)));
      double ans = Math.sqrt(0.636619772 / ax) * (Math.cos(xx) * ans1 - z * Math.sin(xx) * ans2);
      if (x < 0.0) {
        ans = -ans;
      }
      return ans;
    }
  }

  /**
   * Returns the Bessel function of the first kind of order <tt>n</tt> of the argument.
   *
   * @param n the order of the Bessel function.
   * @param x the value to compute the bessel function of.
   */
  public static double jn(int n, double x) {

    if (n == 0) {
      return j0(x);
    }
    if (n == 1) {
      return j1(x);
    }

    double ax = Math.abs(x);
    if (ax == 0.0) {
      return 0.0;
    }

    double ans;
    double tox;
    double bjp;
    double bjm;
    double bj;
    int j;
    if (ax > (double) n) {
      tox = 2.0 / ax;
      bjm = j0(ax);
      bj = j1(ax);
      for (j = 1; j < n; j++) {
        bjp = j * tox * bj - bjm;
        bjm = bj;
        bj = bjp;
      }
      ans = bj;
    } else {
      tox = 2.0 / ax;
      double acc = 40.0;
      int m = 2 * ((n + (int) Math.sqrt(acc * n)) / 2);
      double sum;
      bjp = ans = sum = 0.0;
      bj = 1.0;
      double bigni = 1.0e-10;
      double bigno = 1.0e+10;
      boolean jsum = false;
      for (j = m; j > 0; j--) {
        bjm = j * tox * bj - bjp;
        bjp = bj;
        bj = bjm;
        if (Math.abs(bj) > bigno) {
          bj *= bigni;
          bjp *= bigni;
          ans *= bigni;
          sum *= bigni;
        }
        if (jsum) {
          sum += bj;
        }
        jsum = !jsum;
        if (j == n) {
          ans = bjp;
        }
      }
      sum = 2.0 * sum - bj;
      ans /= sum;
    }
    return x < 0.0 && (n & 0x01) == 1 ? -ans : ans;
  }

  /**
   * Returns the modified Bessel function of the third kind of order 0 of the argument. <p> The range is partitioned
   * into the two intervals [0,8] and (8, infinity).  Chebyshev polynomial expansions are employed in each interval.
   *
   * @param x the value to compute the bessel function of.
   */
  public static double k0(double x) {

    if (x <= 0.0) {
      throw new ArithmeticException();
    }
    if (x <= 2.0) {
      return Arithmetic.chbevl(x * x - 2.0, aSubK0, 10) - Math.log(0.5 * x) * i0(x);
    }

    double z = 8.0 / x - 2.0;
    return Math.exp(-x) * Arithmetic.chbevl(z, bSubK0, 25) / Math.sqrt(x);
  }

  /**
   * Returns the exponentially scaled modified Bessel function of the third kind of order 0 of the argument.
   *
   * @param x the value to compute the bessel function of.
   */
  public static double k0e(double x) {

    if (x <= 0.0) {
      throw new ArithmeticException();
    }
    if (x <= 2.0) {
      return (Arithmetic.chbevl(x * x - 2.0, aSubK0, 10) - Math.log(0.5 * x) * i0(x)) * Math.exp(x);
    }

    return Arithmetic.chbevl(8.0 / x - 2.0, bSubK0, 25) / Math.sqrt(x);
  }

  /**
   * Returns the modified Bessel function of the third kind of order 1 of the argument. <p> The range is partitioned
   * into the two intervals [0,2] and (2, infinity).  Chebyshev polynomial expansions are employed in each interval.
   *
   * @param x the value to compute the bessel function of.
   */
  public static double k1(double x) {

    double z = 0.5 * x;
    if (z <= 0.0) {
      throw new ArithmeticException();
    }
    if (x <= 2.0) {
      return Math.log(z) * i1(x) + Arithmetic.chbevl(x * x - 2.0, aSubK1, 11) / x;
    }

    return Math.exp(-x) * Arithmetic.chbevl(8.0 / x - 2.0, bSubK1, 25) / Math.sqrt(x);
  }

  /**
   * Returns the exponentially scaled modified Bessel function of the third kind of order 1 of the argument. <p>
   * <tt>k1e(x) = exp(x) * k1(x)</tt>.
   *
   * @param x the value to compute the bessel function of.
   */
  public static double k1e(double x) {

    if (x <= 0.0) {
      throw new ArithmeticException();
    }
    if (x <= 2.0) {
      return (Math.log(0.5 * x) * i1(x) + Arithmetic.chbevl(x * x - 2.0, aSubK1, 11) / x) * Math.exp(x);
    }

    return Arithmetic.chbevl(8.0 / x - 2.0, bSubK1, 25) / Math.sqrt(x);
  }

  /**
   * Returns the modified Bessel function of the third kind of order <tt>nn</tt> of the argument. <p> The range is
   * partitioned into the two intervals [0,9.55] and (9.55, infinity).  An ascending power series is used in the low
   * range, and an asymptotic expansion in the high range.
   *
   * @param nn the order of the Bessel function.
   * @param x  the value to compute the bessel function of.
   */
  public static double kn(int nn, double x) {
/*
Algorithm for Kn.
             n-1 
           -n   -  (n-k-1)!    2   k
K (x)  =  0.5 (x/2)     >  -------- (-x /4)
 n                      -     k!
             k=0

          inf.                                   2   k
     n         n   -                                   (x /4)
 + (-1)  0.5(x/2)    >  {p(k+1) + p(n+k+1) - 2log(x/2)} ---------
           -                                  k! (n+k)!
          k=0

where  p(m) is the psi function: p(1) = -EUL and

            m-1
             -
    p(m)  =  -EUL +  >  1/k
             -
            k=1

For large x,
                     2        2     2
                    u-1     (u-1 )(u-3 )
K (z)  =  sqrt(pi/2z) exp(-z) { 1 + ------- + ------------ + ...}
 v                                        1            2
                  1! (8z)     2! (8z)
asymptotically, where

       2
  u = 4 v .

*/


    int n = nn < 0 ? -nn : nn;

    int maxFactorial = 31;
    if (n > maxFactorial) {
      throw new ArithmeticException("Overflow");
    }
    if (x <= 0.0) {
      throw new IllegalArgumentException();
    }

    int i;
    double pk;
    double pn;
    double fn;
    double ans;
    double z;
    double z0;
    double s;
    double t;
    double nk1f;
    double k;
    if (x <= 9.55) {
      ans = 0.0;
      z0 = 0.25 * x * x;
      fn = 1.0;
      pn = 0.0;
      double zmn = 1.0;
      double tox = 2.0 / x;

      if (n > 0) {
        /* compute factorial of n and psi(n) */
        pn = -EULER_GAMMA;
        k = 1.0;
        for (i = 1; i < n; i++) {
          pn += 1.0 / k;
          k += 1.0;
          fn *= k;
        }

        zmn = tox;

        if (n == 1) {
          ans = 1.0 / x;
        } else {
          nk1f = fn / n;
          s = nk1f;
          z = -z0;
          double zn = 1.0;
          double kf = 1.0;
          for (i = 1; i < n; i++) {
            nk1f /= n - i;
            kf *= i;
            zn *= z;
            t = nk1f * zn / kf;
            s += t;
            if ((Double.MAX_VALUE - Math.abs(t)) < Math.abs(s)) {
              throw new ArithmeticException("Overflow");
            }
            if ((tox > 1.0) && ((Double.MAX_VALUE / tox) < zmn)) {
              throw new ArithmeticException("Overflow");
            }
            zmn *= tox;
          }
          s *= 0.5;
          t = Math.abs(s);
          if ((zmn > 1.0) && ((Double.MAX_VALUE / zmn) < t)) {
            throw new ArithmeticException("Overflow");
          }
          if ((t > 1.0) && ((Double.MAX_VALUE / t) < zmn)) {
            throw new ArithmeticException("Overflow");
          }
          ans = s * zmn;
        }
      }


      double tlg = 2.0 * Math.log(0.5 * x);
      pk = -EULER_GAMMA;
      if (n == 0) {
        pn = pk;
        t = 1.0;
      } else {
        pn += 1.0 / n;
        t = 1.0 / fn;
      }
      s = (pk + pn - tlg) * t;
      k = 1.0;
      do {
        t *= z0 / (k * (k + n));
        pk += 1.0 / k;
        pn += 1.0 / (k + n);
        s += (pk + pn - tlg) * t;
        k += 1.0;
      }
      while (Math.abs(t / s) > MACHEP);

      s = 0.5 * s / zmn;
      if ((n & 1) > 0) {
        s = -s;
      }
      ans += s;

      return ans;
    }


    /* Asymptotic expansion for Kn(x) */
    /* Converges to 1.4e-17 for x > 18.4 */
    if (x > MAXLOG) {
      throw new ArithmeticException("Underflow");
    }
    k = n;
    pn = 4.0 * k * k;
    pk = 1.0;
    z0 = 8.0 * x;
    fn = 1.0;
    t = 1.0;
    s = t;
    i = 0;
    double nkf = Double.MAX_VALUE;
    do {
      z = pn - pk * pk;
      t = t * z / (fn * z0);
      nk1f = Math.abs(t);
      if ((i >= n) && (nk1f > nkf)) {
        ans = Math.exp(-x) * Math.sqrt(Math.PI / (2.0 * x)) * s;
        return ans;
      }
      nkf = nk1f;
      s += t;
      fn += 1.0;
      pk += 2.0;
      i += 1;
    } while (Math.abs(t / s) > MACHEP);


    return Math.exp(-x) * Math.sqrt(Math.PI / (2.0 * x)) * s;
  }

  /**
   * Returns the Bessel function of the second kind of order 0 of the argument.
   *
   * @param x the value to compute the bessel function of.
   */
  public static double y0(double x) {
    if (x < 8.0) {
      double y = x * x;
      double ans1 = -2957821389.0 + y * (7062834065.0 + y * (-512359803.6
          + y * (10879881.29 + y * (-86327.92757 + y * 228.4622733))));
      double ans2 = 40076544269.0 + y * (745249964.8 + y * (7189466.438
          + y * (47447.26470 + y * (226.1030244 + y))));

      return (ans1 / ans2) + 0.636619772 * j0(x) * Math.log(x);
    } else {
      double z = 8.0 / x;
      double y = z * z;
      double xx = x - 0.785398164;

      double ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4
          + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
      double ans2 = -0.1562499995e-1 + y * (0.1430488765e-3
          + y * (-0.6911147651e-5 + y * (0.7621095161e-6
          + y * (-0.934945152e-7))));
      return Math.sqrt(0.636619772 / x) * (Math.sin(xx) * ans1 + z * Math.cos(xx) * ans2);
    }
  }

  /**
   * Returns the Bessel function of the second kind of order 1 of the argument.
   *
   * @param x the value to compute the bessel function of.
   */
  public static double y1(double x) {
    if (x < 8.0) {
      double y = x * x;
      double ans1 = x * (-0.4900604943e13 + y * (0.1275274390e13
          + y * (-0.5153438139e11 + y * (0.7349264551e9
          + y * (-0.4237922726e7 + y * 0.8511937935e4)))));
      double ans2 = 0.2499580570e14 + y * (0.4244419664e12
          + y * (0.3733650367e10 + y * (0.2245904002e8
          + y * (0.1020426050e6 + y * (0.3549632885e3 + y)))));
      return (ans1 / ans2) + 0.636619772 * (j1(x) * Math.log(x) - 1.0 / x);
    } else {
      double z = 8.0 / x;
      double y = z * z;
      double xx = x - 2.356194491;
      double ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4
          + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
      double ans2 = 0.04687499995 + y * (-0.2002690873e-3
          + y * (0.8449199096e-5 + y * (-0.88228987e-6
          + y * 0.105787412e-6)));
      return Math.sqrt(0.636619772 / x) * (Math.sin(xx) * ans1 + z * Math.cos(xx) * ans2);
    }
  }

  /**
   * Returns the Bessel function of the second kind of order <tt>n</tt> of the argument.
   *
   * @param n the order of the Bessel function.
   * @param x the value to compute the bessel function of.
   */
  public static double yn(int n, double x) {

    if (n == 0) {
      return y0(x);
    }
    if (n == 1) {
      return y1(x);
    }

    double tox = 2.0 / x;
    double by = y1(x);
    double bym = y0(x);
    for (int j = 1; j < n; j++) {
      double byp = j * tox * by - bym;
      bym = by;
      by = byp;
    }
    return by;
  }
}
