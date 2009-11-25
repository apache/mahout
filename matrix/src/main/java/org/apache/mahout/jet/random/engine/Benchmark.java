/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.jet.random.engine;

/**
 * Benchmarks the performance of the currently provided uniform pseudo-random number generation engines.
 * <p>
 * All distributions are obtained by using a <b>uniform</b> pseudo-random number generation engine.
 * followed by a transformation to the desired distribution.
 * Therefore, the performance of the uniform engines is crucial.
 * <p>
 * <h2 align=center>Comparison of uniform generation engines</h2>
 * <center>
 *   <table border>
 *     <tr> 
 *       <td align="center" width="40%">Name</td>
 *       <td align="center" width="20%">Period</td>
 *       <td align="center" width="40%">
 *         <p>Speed<br>
 *           [# million uniform random numbers generated/sec]<br>
 *           Pentium Pro 200 Mhz, JDK 1.2, NT</p>
 *         </td>
 *     </tr>
 *     <tr> 
 *       <td align="center" width="40%"> <tt>MersenneTwister</tt></td>
 *       <td align="center" width="20%">2<sup>19937</sup>-1 (=10<sup>6001</sup>)</td>
 *       <td align="center" width="40">2.5</td>
 *     </tr>
 *     <tr> 
 *       <td align="center" width="40%"> <tt>Ranlux</tt> (default luxury level 3) </td>
 *       <td align="center" width="20%">10<sup>171</sup></td>
 *       <td align="center" width="40">0.4</td>
 *     </tr>
 *     <tr> 
 *       <td align="center" width="40"> <tt>Ranmar</tt></td>
 *       <td align="center" width="20">10<sup>43</sup></td>
 *       <td align="center" width="40%">1.6</td>
 *     </tr>
 *     <tr> 
 *       <td align="center" width="40%"> <tt>Ranecu</tt> </td>
 *       <td align="center" width="20">10<sup>18</sup></td>
 *       <td align="center" width="40%">1.5</td>
 *     </tr>
 *     <tr> 
 *       <td align="center"> <tt>java.util.Random.nextFloat() </tt><tt> 
 *         </tt></td>
 *       <td align="center"><font size=+3>?</font></td>
 *       <td align="center">2.4</td>
 *     </tr>
 *   </table>
 * </center>
 * <p>
 * <b>Note:</b> Methods working on the default uniform random generator are <b>synchronized</b> and therefore in current VM's <b>slow</b> (as of June '99).
 * Methods taking as argument a uniform random generator are <b>not synchronized</b> and therefore much <b>quicker</b>.
 * Thus, if you need a lot of random numbers, you should use the unsynchronized approach:
 * <p>
 * <b>Example usage:</b><pre>
 * edu.cornell.lassp.houle.RngPack.RandomElement generator;
 * generator = new org.apache.mahout.jet.random.engine.MersenneTwister(new java.util.Date());
 * //generator = new edu.cornell.lassp.houle.RngPack.Ranecu(new java.util.Date());
 * //generator = new edu.cornell.lassp.houle.RngPack.Ranmar(new java.util.Date());
 * //generator = new edu.cornell.lassp.houle.RngPack.Ranlux(new java.util.Date());
 * //generator = makeDefaultGenerator();
 * for (int i=1000000; --i >=0; ) {
 *    double uniform = generator.raw();
 *    ...
 * }
 * </pre>
 *
 *
 * @see org.apache.mahout.jet.random
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class Benchmark {

  private Benchmark() {
  }

  /** Benchmarks <tt>raw()</tt> for various uniform generation engines. */
  public static void benchmark(int times) {
    org.apache.mahout.matrix.Timer timer = new org.apache.mahout.matrix.Timer();

    timer.reset().start();
    for (int i = times; --i >= 0;) {
      ;
    } // no operation
    timer.stop().display();
    float emptyLoop = timer.elapsedTime();
    System.out.println("empty loop timing done.");

    RandomEngine gen = new MersenneTwister();
    System.out.println("\n MersenneTwister:");
    timer.reset().start();
    for (int i = times; --i >= 0;) {
      gen.raw();
    }
    timer.stop().display();
    System.out.println(times / (timer.elapsedTime() - emptyLoop) + " numbers per second.");


    gen = new MersenneTwister64();
    System.out.println("\n MersenneTwister64:");
    timer.reset().start();
    for (int i = times; --i >= 0;) {
      gen.raw();
    }
    timer.stop().display();
    System.out.println(times / (timer.elapsedTime() - emptyLoop) + " numbers per second.");

    /*
    gen = new edu.stanford.mt.MersenneTwister();
    System.out.println("\n edu.stanford.mt.MersenneTwister:");
    timer.reset().start();
    for (int i=times; --i>=0; ) gen.raw();
    timer.stop().display();
    System.out.println(times/(timer.elapsedTime()-emptyLoop)+ " numbers per second.");
    */


    gen = new DRand();
    System.out.println("\nDRand:");
    timer.reset().start();
    for (int i = times; --i >= 0;) {
      gen.raw();
    }
    timer.stop().display();
    System.out.println(times / (timer.elapsedTime() - emptyLoop) + " numbers per second.");


    java.util.Random javaGen = new java.util.Random();
    System.out.println("\njava.util.Random.nextFloat():");
    timer.reset().start();
    for (int i = times; --i >= 0;) {
      javaGen.nextFloat();
    } // nextDouble() is slower
    timer.stop().display();
    System.out.println(times / (timer.elapsedTime() - emptyLoop) + " numbers per second.");

    /*
    gen = new edu.cornell.lassp.houle.RngPack.Ranecu();
    System.out.println("\nRanecu:");
    timer.reset().start();
    for (int i=times; --i>=0; ) gen.raw();
    timer.stop().display();
    System.out.println(times/(timer.elapsedTime()-emptyLoop)+ " numbers per second.");

    gen = new edu.cornell.lassp.houle.RngPack.Ranmar();
    System.out.println("\nRanmar:");
    timer.reset().start();
    for (int i=times; --i>=0; ) gen.raw();
    timer.stop().display();
    System.out.println(times/(timer.elapsedTime()-emptyLoop)+ " numbers per second.");

    gen = new edu.cornell.lassp.houle.RngPack.Ranlux();
    System.out.println("\nRanlux:");
    timer.reset().start();
    for (int i=times; --i>=0; ) gen.raw();
    timer.stop().display();
    System.out.println(times/(timer.elapsedTime()-emptyLoop)+ " numbers per second.");
    */

    System.out.println("\nGood bye.\n");

  }

  /** Tests various methods of this class. */
  public static void main(String[] args) {
    long from = Long.parseLong(args[0]);
    long to = Long.parseLong(args[1]);
    int times = Integer.parseInt(args[2]);
    int runs = Integer.parseInt(args[3]);
    //testRandomFromTo(from,to,times);
    //benchmark(1000000);
    //benchmark(1000000);
    for (int i = 0; i < runs; i++) {
      benchmark(times);
      //benchmarkSync(times);
    }
  }

  /** Prints the first <tt>size</tt> random numbers generated by the given engine. */
  public static void test(int size, RandomEngine randomEngine) {

    /*
    System.out.println("raw():");
    random = (RandomEngine) randomEngine.clone();
    //org.apache.mahout.matrix.Timer timer = new org.apache.mahout.matrix.Timer().start();
    for (int j=0, i=size; --i>=0; j++) {
      System.out.print(" "+random.raw());
      if (j%8==7) System.out.println();
    }

    System.out.println("\n\nfloat():");
    random = (RandomEngine) randomEngine.clone();
    for (int j=0, i=size; --i>=0; j++) {
      System.out.print(" "+random.nextFloat());
      if (j%8==7) System.out.println();
    }

    System.out.println("\n\ndouble():");
    random = (RandomEngine) randomEngine.clone();
    for (int j=0, i=size; --i>=0; j++) {
      System.out.print(" "+random.nextDouble());
      if (j%8==7) System.out.println();
    }
    */
    System.out.println("\n\nint():");
    RandomEngine random = (RandomEngine) randomEngine.clone();
    for (int j = 0, i = size; --i >= 0; j++) {
      System.out.print(" " + random.nextInt());
      if (j % 8 == 7) {
        System.out.println();
      }
    }

    //timer.stop().display();
    System.out.println("\n\nGood bye.\n");
  }

  /** Tests various methods of this class. */
  private static void xtestRandomFromTo(long from, long to, int times) {
    System.out.println("from=" + from + ", to=" + to);

    //org.apache.mahout.matrix.set.OpenMultiFloatHashSet multiset = new org.apache.mahout.matrix.set.OpenMultiFloatHashSet();

    java.util.Random randomJava = new java.util.Random();
    //edu.cornell.lassp.houle.RngPack.RandomElement random = new edu.cornell.lassp.houle.RngPack.Ranecu();
    //edu.cornell.lassp.houle.RngPack.RandomElement random = new edu.cornell.lassp.houle.RngPack.MT19937B();
    //edu.cornell.lassp.houle.RngPack.RandomElement random = new edu.stanford.mt.MersenneTwister();
    RandomEngine random = new MersenneTwister();
    int _from = (int) from, _to = (int) to;
    org.apache.mahout.matrix.Timer timer = new org.apache.mahout.matrix.Timer().start();
    for (int j = 0, i = times; --i >= 0; j++) {
      //randomJava.nextInt(10000);
      //Integers.randomFromTo(_from,_to);
      System.out.print(" " + random.raw());
      if (j % 8 == 7) {
        System.out.println();
      }
      //multiset.add(nextIntFromTo(_from,_to));
    }

    timer.stop().display();
    //System.out.println(multiset); //check the distribution
    System.out.println("Good bye.\n");
  }
}
