package org.apache.mahout.math;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.jet.random.Normal;
import org.junit.Test;

import java.util.Random;

/**
 * Makes sure that a vector under test acts the same as a DenseVector or RandomAccessSparseVector
 * (according to whether it is dense or sparse).  Most operations need to be done within a reasonable
 * tolerance.
 *
 * The idea is that a new vector implementation can extend AbstractVectorTest to get pretty high
 * confidence that it is working correctly.
 */
public abstract class AbstractVectorTest<T extends Vector> extends MahoutTestCase {
  public abstract T vectorToTest(int size);

  @Test
  public void testSimpleOps() {

    final T v0 = vectorToTest(20);
    final Random gen = RandomUtils.getRandom();
    Vector v1 = v0.assign(new Normal(0, 1, gen));
    Vector v2 = vectorToTest(20).assign(new Normal(0, 1, gen));

    assertEquals(v0.get(12), v1.get(12), 0);
    v0.set(12, gen.nextDouble());
    assertEquals(v0.get(12), v1.get(12), 0);
    assertTrue(v0 == v1);

    Vector dv1 = new DenseVector(v1);
    Vector dv2 = new DenseVector(v2);
    Vector sv1 = new RandomAccessSparseVector(v1);
    Vector sv2 = new RandomAccessSparseVector(v2);

    assertEquals(0, dv1.plus(dv2).getDistanceSquared(v1.plus(v2)), 1e-13);
    assertEquals(0, dv1.plus(dv2).getDistanceSquared(v1.plus(dv2)), 1e-13);
    assertEquals(0, dv1.plus(dv2).getDistanceSquared(v1.plus(sv2)), 1e-13);
    assertEquals(0, dv1.plus(dv2).getDistanceSquared(sv1.plus(v2)), 1e-13);

    assertEquals(0, dv1.minus(dv2).getDistanceSquared(v1.minus(v2)), 1e-13);
    assertEquals(0, dv1.minus(dv2).getDistanceSquared(v1.minus(dv2)), 1e-13);
    assertEquals(0, dv1.minus(dv2).getDistanceSquared(v1.minus(sv2)), 1e-13);
    assertEquals(0, dv1.minus(dv2).getDistanceSquared(sv1.minus(v2)), 1e-13);

    double z = gen.nextDouble();
    assertEquals(0, dv1.divide(z).getDistanceSquared(v1.divide(z)), 1e-12);
    assertEquals(0, dv1.times(z).getDistanceSquared(v1.times(z)), 1e-12);
    assertEquals(0, dv1.plus(z).getDistanceSquared(v1.plus(z)), 1e-12);

    assertEquals(dv1.dot(dv2), v1.dot(v2), 1e-13);
    assertEquals(dv1.dot(dv2), v1.dot(dv2), 1e-13);
    assertEquals(dv1.dot(dv2), v1.dot(sv2), 1e-13);
    assertEquals(dv1.dot(dv2), sv1.dot(v2), 1e-13);
    assertEquals(dv1.dot(dv2), dv1.dot(v2), 1e-13);

    assertEquals(dv1.getLengthSquared(), v1.getLengthSquared(), 1e-13);
    assertEquals(dv1.getDistanceSquared(dv2), v1.getDistanceSquared(v2), 1e-13);
    assertEquals(dv1.getDistanceSquared(dv2), dv1.getDistanceSquared(v2), 1e-13);
    assertEquals(dv1.getDistanceSquared(dv2), sv1.getDistanceSquared(v2), 1e-13);
    assertEquals(dv1.getDistanceSquared(dv2), v1.getDistanceSquared(dv2), 1e-13);
    assertEquals(dv1.getDistanceSquared(dv2), v1.getDistanceSquared(sv2), 1e-13);

    assertEquals(dv1.minValue(), v1.minValue(), 1e-13);
    assertEquals(dv1.minValueIndex(), v1.minValueIndex());

    assertEquals(dv1.maxValue(), v1.maxValue(), 1e-13);
    assertEquals(dv1.maxValueIndex(), v1.maxValueIndex());

    Vector nv1 = v1.normalize();

    assertEquals(0, dv1.getDistanceSquared(v1), 1e-13);
    assertEquals(1, nv1.norm(2), 1e-13);
    assertEquals(0, dv1.normalize().getDistanceSquared(nv1), 1e-13);

    nv1 = v1.normalize(1);
    assertEquals(0, dv1.getDistanceSquared(v1), 1e-13);
    assertEquals(1, nv1.norm(1), 1e-13);
    assertEquals(0, dv1.normalize(1).getDistanceSquared(nv1), 1e-13);

    assertEquals(dv1.norm(0), v1.norm(0), 1e-13);
    assertEquals(dv1.norm(1), v1.norm(1), 1e-13);
    assertEquals(dv1.norm(1.5), v1.norm(1.5), 1e-13);
    assertEquals(dv1.norm(2), v1.norm(2), 1e-13);

    // assign double, function, vector x function


    // aggregate

    // cross,

    // getNumNondefaultElements

    for (Vector.Element element : v1) {
      assertEquals(dv1.get(element.index()), element.get(), 0);
      assertEquals(dv1.get(element.index()), v1.get(element.index()), 0);
      assertEquals(dv1.get(element.index()), v1.getQuick(element.index()), 0);
    }
  }
}
