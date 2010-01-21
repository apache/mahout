/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.mahout.math;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import org.apache.mahout.math.function.ByteComparator;
import org.apache.mahout.math.function.CharComparator;
import org.apache.mahout.math.function.DoubleComparator;
import org.apache.mahout.math.function.FloatComparator;
import org.apache.mahout.math.function.IntComparator;
import org.apache.mahout.math.function.LongComparator;
import org.apache.mahout.math.function.ShortComparator;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class SortingTest extends Assert {
  
  private Random random;
  
  private byte[] randomBytes() {
    byte[] bytes = new byte[1000];
    random.nextBytes(bytes);
    return bytes;
  }
  
  private char[] randomChars() {
    char[] chars = new char[100000];
    for (int x = 0; x < 100000; x ++) {
      chars[x] = (char)(random.nextInt() % Character.MAX_VALUE);
    }
    return chars;
  }
  
  private int[] randomInts() {
    int[] ints = new int[100000];
    for (int x = 0; x < 100000; x ++) {
      ints[x] = random.nextInt();
    }
    return ints;
  }

  private short[] randomShorts() {
    short[] shorts = new short[100000];
    for (int x = 0; x < 100000; x ++) {
      shorts[x] = (short)(random.nextInt() % Short.MAX_VALUE);
    }
    return shorts;
  }

  private long[] randomLongs() {
    long[] longs = new long[100000];
    for (int x = 0; x < 100000; x ++) {
      longs[x] = random.nextLong();
    }
    return longs;
  }
  
  private float[] randomFloats() {
    float[] floats = new float[100000];
    for (int x = 0; x < 100000; x ++) {
      floats[x] = random.nextFloat();
    }
    return floats;
  }

  private double[] randomDoubles() {
    double[] doubles = new double[100000];
    for (int x = 0; x < 100000; x ++) {
      doubles[x] = random.nextDouble();
    }
    return doubles;
  }

  @Before
  public void before() {
    random = new Random(0);
  }
  
  static class ForSorting implements Comparable<ForSorting> {
    private final Integer i;
    
    ForSorting(int i) {
      this.i = i;
    }
    
    @Override
    public int compareTo(ForSorting o) {
      return i.compareTo(o.i);
    }
    
    @Override
    public String toString() {
      return i.toString();
    }
  }
  
  static class ReverseCompareForSorting implements Comparator<ForSorting> {
    
    @Override
    public int compare(ForSorting o1, ForSorting o2) {
      return o2.compareTo(o1);
    }
  }
  
  @Test
  public void testBinarySearch() {
    byte[] bytes = {-5, -2, 0, 100, 103};
    int x = Sorting.binarySearchFromTo(bytes, (byte) -6, 0, 4);
    assertEquals(-1, x);
    x = Sorting.binarySearchFromTo(bytes, (byte) 0, 0, 4);
    assertEquals(2, x);
    x = Sorting.binarySearchFromTo(bytes, (byte) 5, 0, 4);
    assertEquals(-4, x);
    x = Sorting.binarySearchFromTo(bytes, (byte) 0, 3, 4);
    assertEquals(-4, x);
    
    char[] chars = {1, 2, 5, 100, 103};
    x = Sorting.binarySearchFromTo(chars, (char) 0, 0, 4);
    assertEquals(-1, x);
    x = Sorting.binarySearchFromTo(chars, (char) 1, 0, 4);
    assertEquals(0, x);
    x = Sorting.binarySearchFromTo(chars, (char) 6, 0, 4);
    assertEquals(-4, x);
    x = Sorting.binarySearchFromTo(chars, (char) 0, 3, 4);
    assertEquals(-4, x);
    
    short[] shorts = {-5, -2, 0, 100, 103};
    x = Sorting.binarySearchFromTo(shorts, (short) -6, 0, 4);
    assertEquals(-1, x);
    x = Sorting.binarySearchFromTo(shorts, (short) 0, 0, 4);
    assertEquals(2, x);
    x = Sorting.binarySearchFromTo(shorts, (short) 5, 0, 4);
    assertEquals(-4, x);
    x = Sorting.binarySearchFromTo(shorts, (short) 0, 3, 4);
    assertEquals(-4, x);
    
    int[] ints = {-5, -2, 0, 100, 103};
    x = Sorting.binarySearchFromTo(ints, (int) -6, 0, 4);
    assertEquals(-1, x);
    x = Sorting.binarySearchFromTo(ints, (int) 0, 0, 4);
    assertEquals(2, x);
    x = Sorting.binarySearchFromTo(ints, (int) 5, 0, 4);
    assertEquals(-4, x);
    x = Sorting.binarySearchFromTo(ints, (int) 0, 3, 4);
    assertEquals(-4, x);
    
    long[] longs = {-5, -2, 0, 100, 103};
    x = Sorting.binarySearchFromTo(longs, (long) -6, 0, 4);
    assertEquals(-1, x);
    x = Sorting.binarySearchFromTo(longs, (long) 0, 0, 4);
    assertEquals(2, x);
    x = Sorting.binarySearchFromTo(longs, (long) 5, 0, 4);
    assertEquals(-4, x);
    x = Sorting.binarySearchFromTo(longs, (long) 0, 3, 4);
    assertEquals(-4, x);
    
    float[] floats = {-5, -2, 0, 100, 103};
    x = Sorting.binarySearchFromTo(floats, (float) -6, 0, 4);
    assertEquals(-1, x);
    x = Sorting.binarySearchFromTo(floats, (float) 0, 0, 4);
    assertEquals(2, x);
    x = Sorting.binarySearchFromTo(floats, (float) 5, 0, 4);
    assertEquals(-4, x);
    x = Sorting.binarySearchFromTo(floats, (float) 0, 3, 4);
    assertEquals(-4, x);
    
    double[] doubles = {-5, -2, 0, 100, 103};
    x = Sorting.binarySearchFromTo(doubles, (double) -6, 0, 4);
    assertEquals(-1, x);
    x = Sorting.binarySearchFromTo(doubles, (double) 0, 0, 4);
    assertEquals(2, x);
    x = Sorting.binarySearchFromTo(doubles, (double) 5, 0, 4);
    assertEquals(-4, x);
    x = Sorting.binarySearchFromTo(doubles, (double) 0, 3, 4);
    assertEquals(-4, x);
  }
  
  @Test
  public void testBinarySearchObjects() {
    List<ForSorting> refList = new ArrayList<ForSorting>();
    refList.add(new ForSorting(-5));
    refList.add(new ForSorting(-2));
    refList.add(new ForSorting(0));
    refList.add(new ForSorting(100));
    refList.add(new ForSorting(103));
    // the compare function is reversed
    Collections.reverse(refList);
    ForSorting[] bsArray = refList.toArray(new ForSorting[5]);
    
    Comparator<ForSorting> comp = new ReverseCompareForSorting();
    
    int x = Sorting.binarySearchFromTo(bsArray, new ForSorting(-6), 0, 4, comp);
    assertEquals(-6, x);
    x = Sorting.binarySearchFromTo(bsArray, new ForSorting(0), 0, 4, comp);
    assertEquals(2, x);
    x = Sorting.binarySearchFromTo(bsArray, new ForSorting(5), 0, 4, comp);
    assertEquals(-3, x);
    x = Sorting.binarySearchFromTo(bsArray, new ForSorting(0), 3, 4, comp);
    assertEquals(-4, x);
  }
  
  @Test
  public void testQuickSortBytes() {
    
    ByteComparator revComp = new ByteComparator() {
      
      @Override
      public int compare(byte o1, byte o2) {
        if (o2 < o1) {
          return -1;
        }
        if (o2 > o1) {
          return 1;
        }
        return 0;
      }
    };
    
    byte[] stuff = randomBytes();
    Sorting.quickSort(stuff, 0, stuff.length, revComp);
    for (int x = 0; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] >= stuff[x + 1]);
    }
    stuff = randomBytes();
    Sorting.quickSort(stuff, 100, stuff.length, revComp);
    for (int x = 100; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] >= stuff[x + 1]);
    }
  }
  
  @Test
  public void testQuickSortChars() {
    char[] stuff = randomChars();
    CharComparator revComp = new CharComparator() {
      
      @Override
      public int compare(char o1, char o2) {
        if (o2 < o1) {
          return -1;
        }
        if (o2 > o1) {
          return 1;
        }
        return 0;
      }
    };
    
    stuff = randomChars();
    Sorting.quickSort(stuff, 0, stuff.length, revComp);
    for (int x = 0; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] >= stuff[x + 1]);
    }
    stuff = randomChars();
    Sorting.quickSort(stuff, 100, stuff.length, revComp);
    for (int x = 100; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] >= stuff[x + 1]);
    }
  }
  
  @Test
  public void testQuickSortInts() {

    IntComparator revComp = new IntComparator() {
      
      @Override
      public int compare(int o1, int o2) {
        if (o2 < o1) {
          return -1;
        }
        if (o2 > o1) {
          return 1;
        }
        return 0;
      }
    };
    
    int[] stuff = randomInts();
    Sorting.quickSort(stuff, 0, stuff.length, revComp);
    for (int x = 0; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] >= stuff[x + 1]);
    }
    stuff = randomInts();
    Sorting.quickSort(stuff, 100, stuff.length, revComp);
    for (int x = 100; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] >= stuff[x + 1]);
    }
  }
  
  @Test
  public void testQuickSortExternals() {
    int[] stuff = randomInts();
    final Integer[] bigInts = new Integer[stuff.length];
    for (int x = 0; x < stuff.length; x ++) {
      bigInts[x] = stuff[x];
    }
    
    Sorting.quickSort(0, stuff.length, new IntComparator() {

      @Override
      public int compare(int o1, int o2) {
        return bigInts[o1].compareTo(bigInts[o2]);
      }}, 
      
      new Swapper() {

        @Override
        public void swap(int a, int b) {
          Integer temp = bigInts[a];
          bigInts[a] = bigInts[b];
          bigInts[b] = temp;
        }});
    
    for (int x = 0; x < (stuff.length - 1); x++) {
      assertTrue("problem at index " + x, bigInts[x].compareTo(bigInts[x + 1]) <= 0);
    }

  }
  
  @Test
  public void testQuickSortLongs() {
    
    LongComparator revComp = new LongComparator() {
      
      @Override
      public int compare(long o1, long o2) {
        if (o2 < o1) {
          return -1;
        }
        if (o2 > o1) {
          return 1;
        }
        return 0;
      }
    };
    
    long[] stuff = randomLongs();
    Sorting.quickSort(stuff, 0, stuff.length, revComp);
    for (int x = 0; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] >= stuff[x + 1]);
    }
  }
  
  @Test
  public void testQuickSortShorts() {
    ShortComparator revComp = new ShortComparator() {
      
      @Override
      public int compare(short o1, short o2) {
        if (o2 < o1) {
          return -1;
        }
        if (o2 > o1) {
          return 1;
        }
        return 0;
      }
    };
    
    short[] stuff = randomShorts();
    Sorting.quickSort(stuff, 0, stuff.length, revComp);
    for (int x = 0; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] >= stuff[x + 1]);
    }
    stuff = randomShorts();
    Sorting.quickSort(stuff, 100, stuff.length, revComp);
    for (int x = 100; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] >= stuff[x + 1]);
    }
  }
  
  @Test
  public void testQuickSortFloats() {
    FloatComparator revComp = new FloatComparator() {
      
      @Override
      public int compare(float o1, float o2) {
        if (o2 < o1) {
          return -1;
        }
        if (o2 > o1) {
          return 1;
        }
        return 0;
      }
    };
    
    float[] stuff = randomFloats();
    Sorting.quickSort(stuff, 0, stuff.length, revComp);
    for (int x = 0; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] >= stuff[x + 1]);
    }
    stuff = randomFloats();
    Sorting.quickSort(stuff, 100, stuff.length, revComp);
    for (int x = 100; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] >= stuff[x + 1]);
    }
  }
  
  @Test
  public void testQuickSortDoubles() {
    DoubleComparator revComp = new DoubleComparator() {
      
      @Override
      public int compare(double o1, double o2) {
        if (o2 < o1) {
          return -1;
        }
        if (o2 > o1) {
          return 1;
        }
        return 0;
      }
    };
    
    double[] stuff = randomDoubles();
    Sorting.quickSort(stuff, 0, stuff.length, revComp);
    for (int x = 0; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] >= stuff[x + 1]);
    }
    stuff = randomDoubles();
    Sorting.quickSort(stuff, 100, stuff.length, revComp);
    for (int x = 100; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] >= stuff[x + 1]);
    }
  }
  
  @Test
  public void testMergeSortBytes() {
    byte[] stuff = randomBytes();
    Sorting.mergeSort(stuff, 0, stuff.length);
    for (int x = 0; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] <= stuff[x + 1]);
    }
    
    stuff = randomBytes();
    Sorting.mergeSort(stuff, 100, stuff.length);
    for (int x = 100; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] <= stuff[x + 1]);
    }
    
    ByteComparator revComp = new ByteComparator() {
      
      @Override
      public int compare(byte o1, byte o2) {
        if (o2 < o1) {
          return -1;
        }
        if (o2 > o1) {
          return 1;
        }
        return 0;
      }
    };
    
    stuff = randomBytes();
    Sorting.mergeSort(stuff, 0, stuff.length, revComp);
    for (int x = 0; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] >= stuff[x + 1]);
    }
  }
  
  @Test
  public void testMergeSortChars() {
    char[] stuff = randomChars();
    Sorting.mergeSort(stuff, 0, stuff.length);
    for (int x = 0; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] <= stuff[x + 1]);
    }
    
    stuff = randomChars();
    Sorting.mergeSort(stuff, 100, stuff.length);
    for (int x = 100; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] <= stuff[x + 1]);
    }
    
    CharComparator revComp = new CharComparator() {
      
      @Override
      public int compare(char o1, char o2) {
        if (o2 < o1) {
          return -1;
        }
        if (o2 > o1) {
          return 1;
        }
        return 0;
      }
    };
    
    stuff = randomChars();
    Sorting.mergeSort(stuff, 0, stuff.length, revComp);
    for (int x = 0; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] >= stuff[x + 1]);
    }
  }
  
  @Test
  public void testMergeSortInts() {
    int[] stuff = randomInts();
    Sorting.mergeSort(stuff, 0, stuff.length);
    for (int x = 0; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] <= stuff[x + 1]);
    }
    
    stuff = randomInts();
    Sorting.mergeSort(stuff, 100, stuff.length);
    for (int x = 100; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] <= stuff[x + 1]);
    }
    
    IntComparator revComp = new IntComparator() {
      
      @Override
      public int compare(int o1, int o2) {
        if (o2 < o1) {
          return -1;
        }
        if (o2 > o1) {
          return 1;
        }
        return 0;
      }
    };
    
    stuff = randomInts();
    Sorting.mergeSort(stuff, 0, stuff.length, revComp);
    for (int x = 0; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] >= stuff[x + 1]);
    }
  }
  
  @Test
  public void testMergeSortLongs() {
    long[] stuff = randomLongs();
    Sorting.mergeSort(stuff, 0, stuff.length);
    for (int x = 0; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] <= stuff[x + 1]);
    }
    
    stuff = randomLongs();
    Sorting.mergeSort(stuff, 100, stuff.length);
    for (int x = 100; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] <= stuff[x + 1]);
    }
    
    LongComparator revComp = new LongComparator() {
      
      @Override
      public int compare(long o1, long o2) {
        if (o2 < o1) {
          return -1;
        }
        if (o2 > o1) {
          return 1;
        }
        return 0;
      }
    };
    
    stuff = randomLongs();
    Sorting.mergeSort(stuff, 0, stuff.length, revComp);
    for (int x = 0; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] >= stuff[x + 1]);
    }
  }
  
  @Test
  public void testMergeSortShorts() {
    short[] stuff = randomShorts();
    Sorting.mergeSort(stuff, 0, stuff.length);
    for (int x = 0; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] <= stuff[x + 1]);
    }
    
    stuff = randomShorts();
    Sorting.mergeSort(stuff, 100, stuff.length);
    for (int x = 100; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] <= stuff[x + 1]);
    }
    
    ShortComparator revComp = new ShortComparator() {
      
      @Override
      public int compare(short o1, short o2) {
        if (o2 < o1) {
          return -1;
        }
        if (o2 > o1) {
          return 1;
        }
        return 0;
      }
    };
    
    stuff = randomShorts();
    Sorting.mergeSort(stuff, 0, stuff.length, revComp);
    for (int x = 0; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] >= stuff[x + 1]);
    }
  }
  
  @Test
  public void testMergeSortFloats() {
    float[] stuff = randomFloats();
    Sorting.mergeSort(stuff, 0, stuff.length);
    for (int x = 0; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] <= stuff[x + 1]);
    }
    
    stuff = randomFloats();
    Sorting.mergeSort(stuff, 100, stuff.length);
    for (int x = 100; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] <= stuff[x + 1]);
    }
    
    FloatComparator revComp = new FloatComparator() {
      
      @Override
      public int compare(float o1, float o2) {
        if (o2 < o1) {
          return -1;
        }
        if (o2 > o1) {
          return 1;
        }
        return 0;
      }
    };
    
    stuff = randomFloats();
    Sorting.mergeSort(stuff, 0, stuff.length, revComp);
    for (int x = 0; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] >= stuff[x + 1]);
    }
  }
  
  @Test
  public void testMergeSortDoubles() {
    double[] stuff = randomDoubles();
    Sorting.mergeSort(stuff, 0, stuff.length);
    for (int x = 0; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] <= stuff[x + 1]);
    }
    
    stuff = randomDoubles();
    Sorting.mergeSort(stuff, 100, stuff.length);
    for (int x = 100; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] <= stuff[x + 1]);
    }
    
    DoubleComparator revComp = new DoubleComparator() {
      
      @Override
      public int compare(double o1, double o2) {
        if (o2 < o1) {
          return -1;
        }
        if (o2 > o1) {
          return 1;
        }
        return 0;
      }
    };
    
    stuff = randomDoubles();
    Sorting.mergeSort(stuff, 0, stuff.length, revComp);
    for (int x = 0; x < (stuff.length - 1); x++) {
      assertTrue(stuff[x] >= stuff[x + 1]);
    }
  }
  

  private static class SomethingToSort implements Swapper, IntComparator {
    private final int[] data;

    private SomethingToSort(int[] data) {
      this.data = data;
    }

    @Override
    public void swap(int a, int b) {
      int temp = data[a];
      data[a] = data[b];
      data[b] = temp;
    }

    @Override
    public int compare(int o1, int o2) {
      if (data[o1] < data[o2]) {
        return -1;
      } else if (data[o1] > data[o2]) {
        return 1;
      } else {
        return 0;
      }
    }
  }

  @Test
  public void testQuickSort() {
    int[] td = new int[20];
    for (int x = 0; x < 20; x ++) {
      td[x] = 20 - x;
    }
    SomethingToSort sts = new SomethingToSort(td);
    Sorting.quickSort(0, 20, sts, sts);
    for (int x = 0; x < 20; x ++) {
      assertEquals(x+1, td[x]);
    }
  }

  private static class SomethingToSortStable implements Swapper, IntComparator {
    private final String[] data;

    private SomethingToSortStable(String[] data) {
      this.data = data;
    }

    @Override
    public void swap(int a, int b) {
      String temp = data[a];
      data[a] = data[b];
      data[b] = temp;
    }

    @Override
    public int compare(int o1, int o2) {
      return data[o1].compareTo(data[o2]);
    }
  }

  @Test
  public void testMergeSort() {
    String[] sd = {new String("z"), new String("a"), new String("a"), new String("q"), new String("1")};
    String[] correct = {sd[4], sd[1], sd[2], sd[3], sd[0]};

    SomethingToSortStable sts = new SomethingToSortStable(sd);
    Sorting.mergeSort(0, 5, sts, sts);

    for (int x = 0; x < 5; x ++) {
      assertSame(correct[x], sd[x]);
    }
  }
  
}

