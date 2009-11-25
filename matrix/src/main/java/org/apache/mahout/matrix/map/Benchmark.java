/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix.map;

import org.apache.mahout.matrix.Timer;
/**
 * Benchmarks the classes of this package.
 *
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class Benchmark {

  private Benchmark() {
  }

  /**
   */
  public static void benchmark(int runs, int size, String kind) {

    System.out.println("initializing...");
    QuickOpenIntIntHashMap map = new QuickOpenIntIntHashMap();

    //for (int i=size; --i >=0; ) {
    for (int i = 0; i < size; i++) {
      map.put(i, i);
    }
    try {
      Thread.sleep(1000);
    } catch (InterruptedException exc) {
    }


    System.out.println("Now benchmarking...");
    int s = 0;
    Timer timer0 = new Timer();
    Timer timer1 = new Timer();
    Timer timer2 = new Timer();
    //map.hashCollisions = 0;
    for (int run = runs; --run >= 0;) {
      if (kind.equals("add")) {
        map.clear();
        //map.ensureCapacity(size*3);
        timer0.start();
        for (int i = size; --i >= 0;) {
          map.put(i, i);
        }
        timer0.stop();
      }
      if (kind.equals("get")) {
        timer0.start();
        for (int i = size; --i >= 0;) {
          s += map.get(i);
        }
        timer0.stop();
      } else {
        timer1.start();
        map.rehash(PrimeFinder.nextPrime(size * 2));
        timer1.stop();

        timer2.start();
        map.rehash(PrimeFinder.nextPrime((int) (size * 1.5)));
        timer2.stop();
      }
    }

    System.out.println("adding: " + timer0);
    System.out.println("growing: " + timer1);
    System.out.println("shrinking: " + timer2);
    System.out.println("total: " + (timer1.plus(timer2)));
    //System.out.println("collisions="+map.hashCollisions);
    System.out.print(s);
  }

  /** Tests various methods of this class. */
  public static void main(String[] args) {
    int runs = Integer.parseInt(args[0]);
    int size = Integer.parseInt(args[1]);
    //boolean add = args[2].equals("add");
    String kind = args[2];
    benchmark(runs, size, kind);
  }

  /**
   */
  public static void test2(int length) {
    org.apache.mahout.jet.random.Uniform uniform =
        new org.apache.mahout.jet.random.Uniform(new org.apache.mahout.jet.random.engine.MersenneTwister());
// using a map
//int[]    keys   = {0    , 3     , 277+3, 277*2+3, 100000, 9    };
//double[] values = {100.0, 1000.0, 277+3, 277*2+3, 70.0  , 71.0 ,};
//int[]    keys   = {0,1,3,4,5,6, 271,272,273,274,275,276,277+5, 277+6,277+7};
    int[] keys = new int[length];
    int to = 10000000;
    for (int i = 0; i < length; i++) {
      keys[i] = uniform.nextIntFromTo(0, to);
    }
    int[] values = keys.clone();

    int size = keys.length;
//AbstractIntIntMap map = new OpenIntIntHashMap(size*2, 0.2, 0.5);
    AbstractIntIntMap map = new OpenIntIntHashMap();

    for (int i = 0; i < keys.length; i++) {
      map.put(keys[i], values[i]);
      //System.out.println(map);
    }

/*
System.out.println(map.containsKey(3));
System.out.println(map.get(3));

System.out.println(map.containsKey(4));
System.out.println(map.get(4));

System.out.println(map.containsValue((int)71.0));
System.out.println(map.keyOf((int)71.0));
*/

//System.out.println(map);
//System.out.println(map.keys());
//System.out.println(map.values());
/*
if (map instanceof QuickOpenIntIntHashMap) {
  System.out.println("totalProbesSaved="+((QuickOpenIntIntHashMap)map).totalProbesSaved);
}
System.out.println("probes="+map.hashCollisions);

map.hashCollisions = 0;
*/
    int sum = 0;
    for (int key : keys) {
      sum += map.get(key);
      //System.out.println(map);
    }
//System.out.println("probes="+map.hashCollisions);

    System.out.println(map);
    System.out.println(sum);
    System.out.println("\n\n");
  }
}
