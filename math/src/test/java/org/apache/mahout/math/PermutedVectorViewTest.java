package org.apache.mahout.math;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.function.DoubleFunction;
import org.junit.Test;

import java.util.Iterator;
import java.util.Random;

public class PermutedVectorViewTest extends MahoutTestCase {
  @Test
  public void testViewBasics() {
    Vector v = randomVector();

    int[] pivot = pivot();

    PermutedVectorView pvv = new PermutedVectorView(v, pivot);

    // verify the view has the same contents
    for (int i = 0; i < 20; i++) {
      assertEquals("Element " + i, v.get(pivot[i]), pvv.get(i), 0);
    }

    // change a view element or two on each side
    pvv.set(6, 321);
    v.set(9, 512);

    // verify again
    for (int i = 0; i < 20; i++) {
      assertEquals("Element " + i, v.get(pivot[i]), pvv.get(i), 0);
    }
  }

  @Test
  public void testIterators() {
    int[] pivot = pivot();
    int[] unpivot = unpivot();

    Vector v = randomVector();
    PermutedVectorView pvv = new PermutedVectorView(v, pivot);

    // check a simple operation and thus an iterator
    assertEquals(v.zSum(), pvv.zSum(), 0);

    assertEquals(v.getNumNondefaultElements(), pvv.getNumNondefaultElements());
    v.set(11, 0);
    assertEquals(v.getNumNondefaultElements(), pvv.getNumNondefaultElements());

    Iterator<Vector.Element> vi = pvv.iterator();
    int i = 0;
    while (vi.hasNext()) {
      Vector.Element e = vi.next();
      assertEquals("Index " + i, i, pivot[e.index()]);
      assertEquals("Reverse Index " + i, unpivot[i], e.index());
      assertEquals("Self-value " + i, e.get(), pvv.get(e.index()), 0);
      // note that we iterate in the original vector order
      assertEquals("Value " + i, v.get(i), e.get(), 0);
      i++;
    }
  }

  private int[] pivot() {
    return new int[]{11, 7, 10, 9, 8, 3, 17, 0, 19, 13, 12, 1, 5, 6, 16, 2, 4, 14, 18, 15};
  }

  private int[] unpivot() {
    int[] pivot = pivot();
    int[] unpivot = new int[20];

    for (int i = 0; i < 20; i++) {
      unpivot[pivot[i]] = i;
    }
    return unpivot;
  }

  private Vector randomVector() {
    Vector v = new DenseVector(20);
    v.assign(new DoubleFunction() {
      Random gen = RandomUtils.getRandom();

      @Override
      public double apply(double arg1) {
        return gen.nextDouble();
      }
    });
    return v;
  }
}
