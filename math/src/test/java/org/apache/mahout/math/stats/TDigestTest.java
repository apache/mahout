/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.math.stats;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.RandomWrapper;
import org.apache.mahout.math.jet.random.AbstractContinousDistribution;
import org.apache.mahout.math.jet.random.Gamma;
import org.apache.mahout.math.jet.random.Normal;
import org.apache.mahout.math.jet.random.Uniform;
import org.junit.*;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.*;

public class TDigestTest {

    private static PrintWriter sizeDump;
    private static PrintWriter errorDump;
    private static PrintWriter deviationDump;

    @BeforeClass
    public static void setup() throws IOException {
        sizeDump = new PrintWriter(new FileWriter("sizes.csv"));
        sizeDump.printf("tag\ti\tq\tk\tactual\n");

        errorDump = new PrintWriter((new FileWriter("errors.csv")));
        errorDump.printf("dist\ttag\tx\tQ\terror\n");

        deviationDump = new PrintWriter((new FileWriter("deviation.csv")));
        deviationDump.printf("tag\tQ\tk\tx\tmean\tleft\tright\tdeviation\n");
    }

    @AfterClass
    public static void teardown() {
        sizeDump.close();
        errorDump.close();
        deviationDump.close();
    }

    @After
    public void flush() {
        sizeDump.flush();
        errorDump.flush();
        deviationDump.flush();
    }

    @Test
    public void testUniform() {
        RandomWrapper gen = RandomUtils.getRandom();
        for (int i = 0; i < 5; i++) {
            runTest(new Uniform(0, 1, gen), 100,
//                    new double[]{0.0001, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999},
                    new double[]{0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999},
                    "uniform", true);
        }
    }

    @Test
    public void testGamma() {
        // this Gamma distribution is very heavily skewed.  The 0.1%-ile is 6.07e-30 while
        // the median is 0.006 and the 99.9th %-ile is 33.6 while the mean is 1.
        // this severe skew means that we have to have positional accuracy that
        // varies by over 11 orders of magnitude.
        RandomWrapper gen = RandomUtils.getRandom();
        for (int i = 0; i < 5; i++) {
            runTest(new Gamma(0.1, 0.1, gen), 100,
//                    new double[]{6.0730483624079e-30, 6.0730483624079e-20, 6.0730483627432e-10, 5.9339110446023e-03,
//                            2.6615455373884e+00, 1.5884778179295e+01, 3.3636770117188e+01},
                    new double[]{0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999},
                    "gamma", true);
        }
    }

    @Test
    public void testNarrowNormal() {
        // this mixture of a uniform and normal distribution has a very narrow peak which is centered
        // near the median.  Our system should be scale invariant and work well regardless.
        final RandomWrapper gen = RandomUtils.getRandom();
        AbstractContinousDistribution mix = new AbstractContinousDistribution() {
            AbstractContinousDistribution normal = new Normal(0, 1e-5, gen);
            AbstractContinousDistribution uniform = new Uniform(-1, 1, gen);

            @Override
            public double nextDouble() {
                double x;
                if (gen.nextDouble() < 0.5) {
                    x = uniform.nextDouble();
                } else {
                    x = normal.nextDouble();
                }
                return x;
            }
        };

        for (int i = 0; i < 5; i++) {
            runTest(mix, 100, new double[]{0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999}, "mixture", false);
        }
    }

    @Test
    public void testRepeatedValues() {
        final RandomWrapper gen = RandomUtils.getRandom();

        // 5% of samples will be 0 or 1.0.  10% for each of the values 0.1 through 0.9
        AbstractContinousDistribution mix = new AbstractContinousDistribution() {
            @Override
            public double nextDouble() {
                return Math.rint(gen.nextDouble() * 10) / 10.0;
            }
        };

        TDigest dist = new TDigest((double) 1000);
        long t0 = System.nanoTime();
        Multiset<Double> data = HashMultiset.create();
        for (int i1 = 0; i1 < 100000; i1++) {
            double x = mix.nextDouble();
            data.add(x);
            dist.add(x);
        }

        System.out.printf("# %fus per point\n", (System.nanoTime() - t0) * 1e-3 / 100000);
        System.out.printf("# %d centroids\n", dist.centroidCount());

        // I would be happier with 5x compression, but repeated values make things kind of weird
        assertTrue("Summary is too large", dist.centroidCount() < 10 * (double) 1000);

        // all quantiles should round to nearest actual value
        for (int i = 0; i < 10; i++) {
            double z = i / 10.0;
            // we skip over troublesome points that are exactly halfway between
            for (double q = z + 0.002; q < z + 0.09; q += 0.005) {
                double cdf = dist.cdf(q);
                // we also relax the tolerances for repeated values
                assertEquals(String.format("z=%.1f, q = %.3f, cdf = %.3f", z, q, cdf), z + 0.05, cdf, 0.005);

                double estimate = dist.quantile(q);
                assertEquals(String.format("z=%.1f, q = %.3f, cdf = %.3f, estimate = %.3f", z, q, cdf, estimate), Math.rint(q * 10) / 10.0, estimate, 0.001);
            }
        }
    }

    @Test
    public void testSequentialPoints() {
        for (int i = 0; i < 5; i++) {
            runTest(new AbstractContinousDistribution() {
                double base = 0;

                @Override
                public double nextDouble() {
                    base += Math.PI * 1e-5;
                    return base;
                }
            }, 100, new double[]{0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999},
                    "sequential", true);
        }
    }

    @Test
    public void testSerialization() {
        Random gen = RandomUtils.getRandom();
        TDigest dist = new TDigest(100);
        for (int i = 0; i < 100000; i++) {
            double x = gen.nextDouble();
            dist.add(x);
        }
        dist.compress();

        ByteBuffer buf = ByteBuffer.allocate(20000);
        dist.asBytes(buf);
        assertTrue(buf.position() < 11000);
        assertEquals(buf.position(), dist.byteSize());
        buf.clear();

        dist.asSmallBytes(buf);
        assertTrue(buf.position() < 6000);
        assertEquals(buf.position(), dist.smallByteSize());

        System.out.printf("# big %d bytes\n", buf.position());

        buf.flip();
        TDigest dist2 = TDigest.fromBytes(buf);
        assertEquals(dist.centroidCount(), dist2.centroidCount());
        assertEquals(dist.compression(), dist2.compression(), 0);
        assertEquals(dist.size(), dist2.size());

        for (double q = 0; q < 1; q += 0.01) {
            assertEquals(dist.quantile(q), dist2.quantile(q), 1e-8);
        }

        Iterator<? extends TDigest.Group> ix = dist2.centroids().iterator();
        for (TDigest.Group group : dist.centroids()) {
            assertTrue(ix.hasNext());
            assertEquals(group.count(), ix.next().count());
        }
        assertFalse(ix.hasNext());

        buf.flip();
        dist.asSmallBytes(buf);
        assertTrue(buf.position() < 6000);
        System.out.printf("# small %d bytes\n", buf.position());

        buf.flip();
        dist2 = TDigest.fromBytes(buf);
        assertEquals(dist.centroidCount(), dist2.centroidCount());
        assertEquals(dist.compression(), dist2.compression(), 0);
        assertEquals(dist.size(), dist2.size());

        for (double q = 0; q < 1; q += 0.01) {
            assertEquals(dist.quantile(q), dist2.quantile(q), 1e-6);
        }

        ix = dist2.centroids().iterator();
        for (TDigest.Group group : dist.centroids()) {
            assertTrue(ix.hasNext());
            assertEquals(group.count(), ix.next().count());
        }
        assertFalse(ix.hasNext());
    }

    @Test
    public void testIntEncoding() {
        Random gen = RandomUtils.getRandom();
        ByteBuffer buf = ByteBuffer.allocate(10000);
        List<Integer> ref = Lists.newArrayList();
        for (int i = 0; i < 3000; i++) {
            int n = gen.nextInt();
            n = n >>> (i / 100);
            ref.add(n);
            TDigest.encode(buf, n);
        }

        buf.flip();

        for (int i = 0; i < 3000; i++) {
            int n = TDigest.decode(buf);
            assertEquals(String.format("%d:", i), ref.get(i).intValue(), n);
        }
    }

    //@Test()
    // very slow running data generator
    public void testSizeControl() {
        RandomWrapper gen = RandomUtils.getRandom();
        System.out.printf("k\tsamples\tcompression\tsize1\tsize2\n");
        for (int k = 0; k < 40; k++) {
            for (int size : new int[]{10, 100, 1000, 10000}) {
                for (double compression : new double[]{2, 5, 10, 20, 50, 100, 200, 500, 1000}) {
                    TDigest dist = new TDigest(compression);
                    for (int i = 0; i < size * 1000; i++) {
                        dist.add(gen.nextDouble());
                    }
                    System.out.printf("%d\t%d\t%.0f\t%d\t%d\n", k, size, compression, dist.smallByteSize(), dist.byteSize());
                }
            }
        }
        System.out.printf("\n");
    }

    @Test
    public void testScaling() {
        RandomWrapper gen = RandomUtils.getRandom();

        System.out.printf("pass\tcompression\tq\terror\tsize\n");
        // change to 50 passes for better graphs
        for (int k = 0; k < 3; k++) {
            List<Double> data = Lists.newArrayList();
            for (int i = 0; i < 100000; i++) {
                data.add(gen.nextDouble());
            }
            Collections.sort(data);

            for (double compression : new double[]{2, 5, 10, 20, 50, 100, 200, 500, 1000}) {
                TDigest dist = new TDigest(compression);
                for (Double x : data) {
                    dist.add(x);
                }
                dist.compress();

                for (double q : new double[]{0.001, 0.01, 0.1, 0.5}) {
                    double estimate = dist.quantile(q);
                    double actual = data.get((int) (q * data.size()));
                    System.out.printf("%d\t%.0f\t%.3f\t%.9f\t%d\n", k, compression, q, estimate - actual, dist.byteSize());
                }
            }
        }
    }

    /**
     * Builds estimates of the CDF of a bunch of data points and checks that the centroids are accurately
     * positioned.  Accuracy is assessed in terms of the estimated CDF which is much more stringent than
     * checking position of quantiles with a single value for desired accuracy.
     *
     * @param gen           Random number generator that generates desired values.
     * @param sizeGuide     Control for size of the histogram.
     * @param tag           Label for the output lines
     * @param recordAllData True if the internal histogrammer should be set up to record all data it sees for
     *                      diagnostic purposes.
     */
    private void runTest(AbstractContinousDistribution gen, double sizeGuide, double[] qValues, String tag, boolean recordAllData) {
        TDigest dist = new TDigest(sizeGuide);
        if (recordAllData) {
            dist.recordAllData();
        }

        long t0 = System.nanoTime();
        List<Double> data = Lists.newArrayList();
        for (int i = 0; i < 100000; i++) {
            double x = gen.nextDouble();
            data.add(x);
            dist.add(x);
        }
        dist.compress();
        Collections.sort(data);

        double[] xValues = qValues.clone();
        for (int i = 0; i < qValues.length; i++) {
            double ix = data.size() * qValues[i] - 0.5;
            int index = (int) Math.floor(ix);
            double p = ix - index;
            xValues[i] = data.get(index) * (1 - p) + data.get(index + 1) * p;
        }

        double qz = 0;
        int iz = 0;
        for (TDigest.Group group : dist.centroids()) {
            double q = (qz + group.count() / 2.0) / dist.size();
            sizeDump.printf("%s\t%d\t%.6f\t%.3f\t%d\n", tag, iz, q, 4 * q * (1 - q) * dist.size() / dist.compression(), group.count());
            qz += group.count();
            iz++;
        }

        System.out.printf("# %fus per point\n", (System.nanoTime() - t0) * 1e-3 / 100000);
        System.out.printf("# %d centroids\n", dist.centroidCount());

        assertTrue("Summary is too large", dist.centroidCount() < 10 * sizeGuide);
        for (int i = 0; i < xValues.length; i++) {
            double x = xValues[i];
            double q = qValues[i];
            double estimate = dist.cdf(x);
            errorDump.printf("%s\t%s\t%.8g\t%.8f\t%.8f\n", tag, "cdf", x, q, estimate - q);
            assertEquals(q, estimate, 0.006);

            estimate = cdf(dist.quantile(q), data);
            errorDump.printf("%s\t%s\t%.8g\t%.8f\t%.8f\n", tag, "quantile", x, q, estimate - q);
            assertEquals(q, estimate, 0.006);
        }

        if (recordAllData) {
            Iterator<? extends TDigest.Group> ix = dist.centroids().iterator();
            TDigest.Group b = ix.next();
            TDigest.Group c = ix.next();
            qz = b.count();
            while (ix.hasNext()) {
                TDigest.Group a = b;
                b = c;
                c = ix.next();
                double left = (b.mean() - a.mean()) / 2;
                double right = (c.mean() - b.mean()) / 2;

                double q = (qz + b.count() / 2.0) / dist.size();
                for (Double x : b.data()) {
                    deviationDump.printf("%s\t%.5f\t%d\t%.5g\t%.5g\t%.5g\t%.5g\t%.5f\n", tag, q, b.count(), x, b.mean(), left, right, (x - b.mean()) / (right + left));
                }
                qz += a.count();
            }
        }
    }

    @Test
    public void testMerge() {
        RandomWrapper gen = RandomUtils.getRandom();

        for (int parts : new int[]{2, 5, 10, 20, 50, 100}) {
            List<Double> data = Lists.newArrayList();

            TDigest dist = new TDigest(100);
            dist.recordAllData();

            List<TDigest> many = Lists.newArrayList();
            for (int i = 0; i < 100; i++) {
                many.add(new TDigest(100).recordAllData());
            }

            // we accumulate the data into multiple sub-digests
            List<TDigest> subs = Lists.newArrayList();
            for (int i = 0; i < parts; i++) {
                subs.add(new TDigest(50).recordAllData());
            }

            for (int i = 0; i < 100000; i++) {
                double x = gen.nextDouble();
                data.add(x);
                dist.add(x);
                subs.get(i % parts).add(x);
            }
            dist.compress();
            Collections.sort(data);

            // collect the raw data from the sub-digests
            List<Double> data2 = Lists.newArrayList();
            for (TDigest digest : subs) {
                for (TDigest.Group group : digest.centroids()) {
                    Iterables.addAll(data2, group.data());
                }
            }
            Collections.sort(data2);

            // verify that the raw data all got recorded
            assertEquals(data.size(), data2.size());
            Iterator<Double> ix = data.iterator();
            for (Double x : data2) {
                assertEquals(ix.next(), x);
            }

            // now merge the sub-digests
            TDigest dist2 = TDigest.merge(50, subs);

            for (double q : new double[]{0.001, 0.01, 0.1, 0.2, 0.3, 0.5}) {
                double z = quantile(q, data);
                double e1 = dist.quantile(q) - z;
                double e2 = dist2.quantile(q) - z;
                System.out.printf("quantile\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n", parts, q, z - q, e1, e2, Math.abs(e2) / q);
                assertTrue(String.format("parts=%d, q=%.4f, e1=%.5f, e2=%.5f, rel=%.4f", parts, q, e1, e2, Math.abs(e2) / q), Math.abs(e2) / q < 0.1 && Math.abs(e2) < 0.01);
            }

            for (double x : new double[]{0.001, 0.01, 0.1, 0.2, 0.3, 0.5}) {
                double z = cdf(x, data);
                double e1 = dist.cdf(x) - z;
                double e2 = dist2.cdf(x) - z;

                System.out.printf("cdf\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n", parts, x, z - x, e1, e2, Math.abs(e2) / x);
                assertTrue(String.format("parts=%d, x=%.4f, e1=%.5f, e2=%.5f", parts, x, e1, e2), Math.abs(e2) / x < 0.1 && Math.abs(e2) < 0.01);
            }
        }
    }

    private double cdf(final double x, List<Double> data) {
        int n1 = 0;
        int n2 = 0;
        for (Double v : data) {
            n1 += (v < x) ? 1 : 0;
            n2 += (v <= x) ? 1 : 0;
        }
        return (n1 + n2) / 2.0 / data.size();
    }

    private double quantile(final double q, List<Double> data) {
        return data.get((int) Math.floor(data.size() * q));
    }

    @Before
    public void setUp() {
        RandomUtils.useTestSeed();
    }
}
