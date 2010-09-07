/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.random;

import java.util.Random;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class EmpiricalWalker extends AbstractDiscreteDistribution {

  private int K;
  private int[] A;
  private double[] F;

  private double[] cdf; // cumulative distribution function

  /*
  * James Theiler, jt@lanl.gov, the author of the GSL routine this port is based on, describes his approach as follows:
  *
  * Based on: Alastair J Walker, An efficient method for generating
  * discrete random variables with general distributions, ACM Trans
  * Math Soft 3, 253-256 (1977).  See also: D. E. Knuth, The Art of
  * Computer Programming, Volume 2 (Seminumerical algorithms), 3rd
  * edition, Addison-Wesley (1997), p120.

  * Walker's algorithm does some preprocessing, and provides two
  * arrays: floating point F[k] and integer A[k].  A value k is chosen
  * from 0..K-1 with equal likelihood, and then a uniform random number
  * u is compared to F[k].  If it is less than F[k], then k is
  * returned.  Otherwise, A[k] is returned.

  * Walker's original paper describes an O(K^2) algorithm for setting
  * up the F and A arrays.  I found this disturbing since I wanted to
  * use very large values of K.  I'm sure I'm not the first to realize
  * this, but in fact the preprocessing can be done in O(K) steps.

  * A figure of merit for the preprocessing is the average value for
  * the F[k]'s (that is, SUM_k F[k]/K); this corresponds to the
  * probability that k is returned, instead of A[k], thereby saving a
  * redirection.  Walker's O(K^2) preprocessing will generally improve
  * that figure of merit, compared to my cheaper O(K) method; from some
  * experiments with a perl script, I get values of around 0.6 for my
  * method and just under 0.75 for Walker's.  Knuth has pointed out
  * that finding _the_ optimum lookup tables, which maximize the
  * average F[k], is a combinatorially difficult problem.  But any
  * valid preprocessing will still provide O(1) time for the call to
  * gsl_ran_discrete().  I find that if I artificially set F[k]=1 --
  * ie, better than optimum! -- I get a speedup of maybe 20%, so that's
  * the maximum I could expect from the most expensive preprocessing.
  * Folding in the difference of 0.6 vs 0.75, I'd estimate that the
  * speedup would be less than 10%.

  * I've not implemented it here, but one compromise is to sort the
  * probabilities once, and then work from the two ends inward.  This
  * requires O(K log K), still lots cheaper than O(K^2), and from my
  * experiments with the perl script, the figure of merit is within
  * about 0.01 for K up to 1000, and no sign of diverging (in fact,
  * they seemed to be converging, but it's hard to say with just a
  * handful of runs).

  * The O(K) algorithm goes through all the p_k's and decides if they
  * are "smalls" or "bigs" according to whether they are less than or
  * greater than the mean value 1/K.  The indices to the smalls and the
  * bigs are put in separate stacks, and then we work through the
  * stacks together.  For each small, we pair it up with the next big
  * in the stack (Walker always wanted to pair up the smallest small
  * with the biggest big).  The small "borrows" from the big just
  * enough to bring the small up to mean.  This reduces the size of the
  * big, so the (smaller) big is compared again to the mean, and if it
  * is smaller, it gets "popped" from the big stack and "pushed" to the
  * small stack.  Otherwise, it stays put.  Since every time we pop a
  * small, we are able to deal with it right then and there, and we
  * never have to pop more than K smalls, then the algorithm is O(K).

  * This implementation sets up two separate stacks, and allocates K
  * elements between them.  Since neither stack ever grows, we do an
  * extra O(K) pass through the data to determine how many smalls and
  * bigs there are to begin with and allocate appropriately.  In all
  * there are 2*K*sizeof(double) transient bytes of memory that are
  * used than returned, and K*(sizeof(int)+sizeof(double)) bytes used
  * in the lookup table.

  * Walker spoke of using two random numbers (an integer 0..K-1, and a
  * floating point u in [0,1]), but Knuth points out that one can just
  * use the integer and fractional parts of K*u where u is in [0,1].
  * In fact, Knuth further notes that taking F'[k]=(k+F[k])/K, one can
  * directly compare u to F'[k] without having to explicitly set
  * u=K*u-int(K*u).

  * Usage:

  * Starting with an array of probabilities P, initialize and do
  * preprocessing with a call to:

  *    gsl_rng *r;
  *    gsl_ran_discrete_t *f;
  *    f = gsl_ran_discrete_preproc(K,P);

  * Then, whenever a random index 0..K-1 is desired, use

  *    k = gsl_ran_discrete(r,f);

  * Note that several different randevent struct's can be
  * simultaneously active.

  * Aside: A very clever alternative approach is described in
  * Abramowitz and Stegun, p 950, citing: Marsaglia, Random variables
  * and computers, Proc Third Prague Conference in Probability Theory,
  * 1962.  A more accesible reference is: G. Marsaglia, Generating
  * discrete random numbers in a computer, Comm ACM 6, 37-38 (1963).
  * If anybody is interested, I (jt) have also coded up this version as
  * part of another software package.  However, I've done some
  * comparisons, and the Walker method is both faster and more stingy
  * with memory.  So, in the end I decided not to include it with the
  * GSL package.

  * Written 26 Jan 1999, James Theiler, jt@lanl.gov
  * Adapted to GSL, 30 Jan 1999, jt

  */

  /**
   * Constructs an Empirical distribution. The probability distribution function (pdf) is an array of positive real
   * numbers. It need not be provided in the form of relative probabilities, absolute probabilities are also accepted.
   * The <tt>pdf</tt> must satisfy both of the following conditions <ul> <li><tt>0.0 &lt;= pdf[i] :
   * 0&lt;=i&lt;=pdf.length-1</tt> <li><tt>0.0 &lt; Sum(pdf[i]) : 0&lt;=i&lt;=pdf.length-1</tt> </ul>
   *
   * @param pdf               the probability distribution function.
   * @param interpolationType can be either
   *  <tt>Empirical.NO_INTERPOLATION</tt> or <tt>Empirical.LINEAR_INTERPOLATION</tt>.
   * @param randomGenerator   a uniform random number generator.
   * @throws IllegalArgumentException if at least one of the three conditions above is violated.
   */
  public EmpiricalWalker(double[] pdf, int interpolationType, Random randomGenerator) {
    setRandomGenerator(randomGenerator);
    setState(pdf, interpolationType);
    setState2(pdf);
  }

  /** Returns the cumulative distribution function. */
  public double cdf(int k) {
    if (k < 0) {
      return 0.0;
    }
    if (k >= cdf.length - 1) {
      return 1.0;
    }
    return cdf[k];
  }

  /**
   * Returns a deep copy of the receiver; the copy will produce identical sequences. After this call has returned, the
   * copy and the receiver have equal but separate state.
   *
   * @return a copy of the receiver.
   */
  @Override
  public Object clone() {
    EmpiricalWalker copy = (EmpiricalWalker) super.clone();
    if (this.cdf != null) {
      copy.cdf = this.cdf.clone();
    }
    if (this.A != null) {
      copy.A = this.A.clone();
    }
    if (this.F != null) {
      copy.F = this.F.clone();
    }
    return copy;
  }

  /** Returns a random integer <tt>k</tt> with probability <tt>pdf(k)</tt>.     */
  @Override
  public int nextInt() {
    double u = this.randomGenerator.nextDouble();
    u *= this.K;
    int c = (int) u;
    u -= c;
    double f = this.F[c];
    if (f == 1.0) {
      return c;
    }
    return u < f ? c : this.A[c];
  }

  /** Returns the probability distribution function. */
  public double pdf(int k) {
    if (k < 0 || k >= cdf.length - 1) {
      return 0.0;
    }
    return cdf[k - 1] - cdf[k];
  }

  /**
   * Sets the distribution parameters. The <tt>pdf</tt> must satisfy all of the following conditions <ul> <li><tt>pdf !=
   * null && pdf.length &gt; 0</tt> <li><tt>0.0 &lt;= pdf[i] : 0 &lt; =i &lt;= pdf.length-1</tt> <li><tt>0.0 &lt;
   * Sum(pdf[i]) : 0 &lt;=i &lt;= pdf.length-1</tt> </ul>
   *
   * @param pdf probability distribution function.
   * @throws IllegalArgumentException if at least one of the three conditions above is violated.
   */
  public void setState(double[] pdf, int interpolationType) {
    if (pdf == null || pdf.length == 0) {
      throw new IllegalArgumentException("Non-existing pdf");
    }

    // compute cumulative distribution function (cdf) from probability distribution function (pdf)
    int nBins = pdf.length;
    this.cdf = new double[nBins + 1];

    cdf[0] = 0;
    for (int i = 0; i < nBins; i++) {
      if (pdf[i] < 0.0) {
        throw new IllegalArgumentException("Negative probability");
      }
      cdf[i + 1] = cdf[i] + pdf[i];
    }
    if (cdf[nBins] <= 0.0) {
      throw new IllegalArgumentException("At leat one probability must be > 0.0");
    }
    // now normalize to 1 (relative probabilities).
    for (int i = 0; i < nBins + 1; i++) {
      cdf[i] /= cdf[nBins];
    }
    // cdf is now cached...
  }

  /**
   * Sets the distribution parameters. The <tt>pdf</tt> must satisfy both of the following conditions <ul> <li><tt>0.0
   * &lt;= pdf[i] : 0 &lt; =i &lt;= pdf.length-1</tt> <li><tt>0.0 &lt; Sum(pdf[i]) : 0 &lt;=i &lt;= pdf.length-1</tt>
   * </ul>
   *
   * @param pdf probability distribution function.
   * @throws IllegalArgumentException if at least one of the three conditions above is violated.
   */
  public void setState2(double[] pdf) {
    int size = pdf.length;
    int k;
    double pTotal = 0;

    //if (size < 1) {
    //  throw new IllegalArgumentException("must have size greater than zero");
    //}
    /* Make sure elements of ProbArray[] are positive.
    * Won't enforce that sum is unity; instead will just normalize
    */
    for (k = 0; k < size; ++k) {
      //if (pdf[k] < 0) {
      //throw new IllegalArgumentException("Probabilities must be >= 0: "+pdf[k]);
      //}
      pTotal += pdf[k];
    }

    /* Begin setting up the internal state */
    this.K = size;
    this.F = new double[size];
    this.A = new int[size];

    // normalize to relative probability
    double[] E = new double[size];
    for (k = 0; k < size; ++k) {
      E[k] = pdf[k] / pTotal;
    }

    /* Now create the Bigs and the Smalls */
    double mean = 1.0 / size;
    int nSmalls = 0;
    int nBigs = 0;
    for (k = 0; k < size; ++k) {
      if (E[k] < mean) {
        ++nSmalls;
      } else {
        ++nBigs;
      }
    }
    Stack bigs = new Stack(nBigs);
    Stack smalls = new Stack(nSmalls);
    for (k = 0; k < size; ++k) {
      if (E[k] < mean) {
        smalls.push(k);
      } else {
        bigs.push(k);
      }
    }
    /* Now work through the smalls */
    int b;
    while (smalls.size() > 0) {
      int s = smalls.pop();
      if (bigs.size() == 0) {
        /* Then we are on our last value */
        this.A[s] = s;
        this.F[s] = 1.0;
        break;
      }
      b = bigs.pop();
      this.A[s] = b;
      this.F[s] = size * E[s];

      double d = mean - E[s];
      E[s] += d;              /* now E[s] == mean */
      E[b] -= d;
      if (E[b] < mean) {
        smalls.push(b); /* no longer big, join ranks of the small */
      } else if (E[b] > mean) {
        bigs.push(b); /* still big, put it back where you found it */
      } else {
        /* E[b]==mean implies it is finished too */
        this.A[b] = b;
        this.F[b] = 1.0;
      }
    }
    while (bigs.size() > 0) {
      b = bigs.pop();
      this.A[b] = b;
      this.F[b] = 1.0;
    }
  }

  /** Returns a String representation of the receiver. */
  public String toString() {
    return this.getClass().getName() + '(' + ((cdf != null) ? cdf.length : 0) + ')';
  }
}
