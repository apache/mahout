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

package org.apache.mahout.math.random;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.DiagonalMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;

import java.util.Random;

/**
 * Samples from a multi-variate normal distribution.
 */
public class MultiNormal implements Sampler<Vector> {
    private final Random gen;
    private final int dimension;
    private final Matrix mean;
    private final Vector offset;

    public MultiNormal(Vector diagonal) {
        this(new DiagonalMatrix(diagonal), null);
    }

    public MultiNormal(Vector diagonal, Vector offset) {
        this(new DiagonalMatrix(diagonal), offset);
    }

    public MultiNormal(Matrix a, Vector offset) {
        this(a, offset, a.columnSize());
    }

    public MultiNormal(int dimension) {
        this(null, null, dimension);
    }

    public MultiNormal(double radius, Vector mean) {
        this(new DiagonalMatrix(radius, mean.size()), mean);
    }

    private MultiNormal(Matrix mean, Vector offset, int dimension) {
        gen = RandomUtils.getRandom();
        this.dimension = dimension;
        this.mean = mean;
        this.offset = offset;
    }

    @Override
    public Vector sample() {
        Vector v = new DenseVector(dimension).assign(
                new DoubleFunction() {
                    @Override
                    public double apply(double ignored) {
                        return gen.nextGaussian();
                    }
                }
        );
        if (offset != null) {
            return mean.times(v).plus(offset);
        } else {
            if (mean != null) {
                return mean.times(v);
            } else {
                return v;
            }
        }
    }

    public Vector getMean() {
        return offset;
    }
}
