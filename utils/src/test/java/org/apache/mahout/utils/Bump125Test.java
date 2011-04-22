package org.apache.mahout.utils;

import com.google.common.collect.Lists;
import org.junit.Test;

import java.util.Iterator;

public class Bump125Test extends MahoutTestCase {
  @Test
  public void testIncrement() throws Exception {
    Iterator<Integer> ref = Lists.newArrayList(1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60,
            70, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350,
            400, 500, 600, 700, 800, 1000, 1200, 1400, 1600, 1800,
            2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000)
            .iterator();
    Bump125 b = new Bump125();
    for (int i = 0; i < 50; i++) {
      final long x = b.increment();
      assertEquals(ref.next().longValue(), x);
    }
  }
}
