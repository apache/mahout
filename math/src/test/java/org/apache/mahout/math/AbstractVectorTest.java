package org.apache.mahout.math;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.function.Functions;
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

  private static final double FUZZ = 1.0e-13;

  public abstract T vectorToTest(int size);

  @Test
  public void testSimpleOps() {

    T v0 = vectorToTest(20);
    Random gen = RandomUtils.getRandom();
    Vector v1 = v0.assign(new Normal(0, 1, gen));
    Vector v2 = vectorToTest(20).assign(new Normal(0, 1, gen));

    assertEquals(v0.get(12), v1.get(12), 0);
    v0.set(12, gen.nextDouble());
    assertEquals(v0.get(12), v1.get(12), 0);
    assertSame(v0, v1);

    Vector dv1 = new DenseVector(v1);
    Vector dv2 = new DenseVector(v2);
    Vector sv1 = new RandomAccessSparseVector(v1);
    Vector sv2 = new RandomAccessSparseVector(v2);

    assertEquals(0, dv1.plus(dv2).getDistanceSquared(v1.plus(v2)), FUZZ);
    assertEquals(0, dv1.plus(dv2).getDistanceSquared(v1.plus(dv2)), FUZZ);
    assertEquals(0, dv1.plus(dv2).getDistanceSquared(v1.plus(sv2)), FUZZ);
    assertEquals(0, dv1.plus(dv2).getDistanceSquared(sv1.plus(v2)), FUZZ);

    assertEquals(0, dv1.times(dv2).getDistanceSquared(v1.times(v2)), FUZZ);
    assertEquals(0, dv1.times(dv2).getDistanceSquared(v1.times(dv2)), FUZZ);
    assertEquals(0, dv1.times(dv2).getDistanceSquared(v1.times(sv2)), FUZZ);
    assertEquals(0, dv1.times(dv2).getDistanceSquared(sv1.times(v2)), FUZZ);

    assertEquals(0, dv1.minus(dv2).getDistanceSquared(v1.minus(v2)), FUZZ);
    assertEquals(0, dv1.minus(dv2).getDistanceSquared(v1.minus(dv2)), FUZZ);
    assertEquals(0, dv1.minus(dv2).getDistanceSquared(v1.minus(sv2)), FUZZ);
    assertEquals(0, dv1.minus(dv2).getDistanceSquared(sv1.minus(v2)), FUZZ);

    double z = gen.nextDouble();
    assertEquals(0, dv1.divide(z).getDistanceSquared(v1.divide(z)), 1.0e-12);
    assertEquals(0, dv1.times(z).getDistanceSquared(v1.times(z)), 1.0e-12);
    assertEquals(0, dv1.plus(z).getDistanceSquared(v1.plus(z)), 1.0e-12);

    assertEquals(dv1.dot(dv2), v1.dot(v2), FUZZ);
    assertEquals(dv1.dot(dv2), v1.dot(dv2), FUZZ);
    assertEquals(dv1.dot(dv2), v1.dot(sv2), FUZZ);
    assertEquals(dv1.dot(dv2), sv1.dot(v2), FUZZ);
    assertEquals(dv1.dot(dv2), dv1.dot(v2), FUZZ);

    // first attempt has no cached distances
    assertEquals(dv1.getDistanceSquared(dv2), v1.getDistanceSquared(v2), FUZZ);
    assertEquals(dv1.getDistanceSquared(dv2), dv1.getDistanceSquared(v2), FUZZ);
    assertEquals(dv1.getDistanceSquared(dv2), sv1.getDistanceSquared(v2), FUZZ);
    assertEquals(dv1.getDistanceSquared(dv2), v1.getDistanceSquared(dv2), FUZZ);
    assertEquals(dv1.getDistanceSquared(dv2), v1.getDistanceSquared(sv2), FUZZ);

    // now repeat with cached sizes
    assertEquals(dv1.getLengthSquared(), v1.getLengthSquared(), FUZZ);
    assertEquals(dv1.getDistanceSquared(dv2), v1.getDistanceSquared(v2), FUZZ);
    assertEquals(dv1.getDistanceSquared(dv2), dv1.getDistanceSquared(v2), FUZZ);
    assertEquals(dv1.getDistanceSquared(dv2), sv1.getDistanceSquared(v2), FUZZ);
    assertEquals(dv1.getDistanceSquared(dv2), v1.getDistanceSquared(dv2), FUZZ);
    assertEquals(dv1.getDistanceSquared(dv2), v1.getDistanceSquared(sv2), FUZZ);

    assertEquals(dv1.minValue(), v1.minValue(), FUZZ);
    assertEquals(dv1.minValueIndex(), v1.minValueIndex());

    assertEquals(dv1.maxValue(), v1.maxValue(), FUZZ);
    assertEquals(dv1.maxValueIndex(), v1.maxValueIndex());

    Vector nv1 = v1.normalize();

    assertEquals(0, dv1.getDistanceSquared(v1), FUZZ);
    assertEquals(1, nv1.norm(2), FUZZ);
    assertEquals(0, dv1.normalize().getDistanceSquared(nv1), FUZZ);

    nv1 = v1.normalize(1);
    assertEquals(0, dv1.getDistanceSquared(v1), FUZZ);
    assertEquals(1, nv1.norm(1), FUZZ);
    assertEquals(0, dv1.normalize(1).getDistanceSquared(nv1), FUZZ);

    assertEquals(dv1.norm(0), v1.norm(0), FUZZ);
    assertEquals(dv1.norm(1), v1.norm(1), FUZZ);
    assertEquals(dv1.norm(1.5), v1.norm(1.5), FUZZ);
    assertEquals(dv1.norm(2), v1.norm(2), FUZZ);

    assertEquals(dv1.zSum(), v1.zSum(), FUZZ);

    assertEquals(3.1 * v1.size(), v1.assign(3.1).zSum(), FUZZ);
    assertEquals(0, v1.plus(-3.1).norm(1), FUZZ);
    v1.assign(dv1);
    assertEquals(0, v1.getDistanceSquared(dv1), FUZZ);

    assertEquals(dv1.zSum() - dv1.size() * 3.4, v1.assign(Functions.minus(3.4)).zSum(), FUZZ);
    assertEquals(dv1.zSum() - dv1.size() * 4.5, v1.assign(Functions.MINUS, 1.1).zSum(), FUZZ);
    v1.assign(dv1);

    assertEquals(0, dv1.minus(dv2).getDistanceSquared(v1.assign(v2, Functions.MINUS)), FUZZ);
    v1.assign(dv1);

    assertEquals(dv1.norm(2), Math.sqrt(v1.aggregate(Functions.PLUS, Functions.pow(2))), FUZZ);
    assertEquals(dv1.dot(dv2), v1.aggregate(v2, Functions.PLUS, Functions.MULT), FUZZ);

    assertEquals(dv1.viewPart(5, 10).zSum(), v1.viewPart(5, 10).zSum(), FUZZ);

    Vector v3 = v1.clone();
    assertEquals(0, v1.getDistanceSquared(v3), FUZZ);
    assertNotSame(v1, v3);
    v3.assign(0);
    assertEquals(0, dv1.getDistanceSquared(v1), FUZZ);
    assertEquals(0, v3.getLengthSquared(), FUZZ);

    dv1.assign(Functions.ABS);
    v1.assign(Functions.ABS);
    assertEquals(0, dv1.logNormalize().getDistanceSquared(v1.logNormalize()), FUZZ);
    assertEquals(0, dv1.logNormalize(1.5).getDistanceSquared(v1.logNormalize(1.5)), FUZZ);

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
