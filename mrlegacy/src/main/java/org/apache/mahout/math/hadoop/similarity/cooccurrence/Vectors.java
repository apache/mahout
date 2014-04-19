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

package org.apache.mahout.math.hadoop.similarity.cooccurrence;

import java.io.DataInput;
import java.io.IOException;
import java.util.Iterator;

import com.google.common.base.Preconditions;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.iterator.FixedSizeSamplingIterator;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Varint;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.map.OpenIntIntHashMap;

public final class Vectors {

  private Vectors() {}

  public static Vector maybeSample(Vector original, int sampleSize) {
    if (original.getNumNondefaultElements() <= sampleSize) {
      return original;
    }
    Vector sample = new RandomAccessSparseVector(original.size(), sampleSize);
    Iterator<Element> sampledElements =
        new FixedSizeSamplingIterator<Vector.Element>(sampleSize, original.nonZeroes().iterator());
    while (sampledElements.hasNext()) {
      Element elem = sampledElements.next();
      sample.setQuick(elem.index(), elem.get());
    }
    return sample;
  }

  public static Vector topKElements(int k, Vector original) {
    if (original.getNumNondefaultElements() <= k) {
      return original;
    }

    TopElementsQueue topKQueue = new TopElementsQueue(k);
    for (Element nonZeroElement : original.nonZeroes()) {
      MutableElement top = topKQueue.top();
      double candidateValue = nonZeroElement.get();
      if (candidateValue > top.get()) {
        top.setIndex(nonZeroElement.index());
        top.set(candidateValue);
        topKQueue.updateTop();
      }
    }

    Vector topKSimilarities = new RandomAccessSparseVector(original.size(), k);
    for (Vector.Element topKSimilarity : topKQueue.getTopElements()) {
      topKSimilarities.setQuick(topKSimilarity.index(), topKSimilarity.get());
    }
    return topKSimilarities;
  }

  public static Vector merge(Iterable<VectorWritable> partialVectors) {
    Iterator<VectorWritable> vectors = partialVectors.iterator();
    Vector accumulator = vectors.next().get();
    while (vectors.hasNext()) {
      VectorWritable v = vectors.next();
      if (v != null) {
        for (Element nonZeroElement : v.get().nonZeroes()) {
          accumulator.setQuick(nonZeroElement.index(), nonZeroElement.get());
        }
      }
    }
    return accumulator;
  }

  public static Vector sum(Iterator<VectorWritable> vectors) {
    Vector sum = vectors.next().get();
    while (vectors.hasNext()) {
      sum.assign(vectors.next().get(), Functions.PLUS);
    }
    return sum;
  }

  static class TemporaryElement implements Vector.Element {

    private final int index;
    private double value;

    TemporaryElement(int index, double value) {
      this.index = index;
      this.value = value;
    }

    TemporaryElement(Vector.Element toClone) {
      this(toClone.index(), toClone.get());
    }

    @Override
    public double get() {
      return value;
    }

    @Override
    public int index() {
      return index;
    }

    @Override
    public void set(double value) {
      this.value = value;
    }
  }

  public static Vector.Element[] toArray(VectorWritable vectorWritable) {
    Vector.Element[] elements = new Vector.Element[vectorWritable.get().getNumNondefaultElements()];
    int k = 0;
    for (Element nonZeroElement : vectorWritable.get().nonZeroes()) {
      elements[k++] = new TemporaryElement(nonZeroElement.index(), nonZeroElement.get());
    }
    return elements;
  }

  public static void write(Vector vector, Path path, Configuration conf) throws IOException {
    write(vector, path, conf, false);
  }

  public static void write(Vector vector, Path path, Configuration conf, boolean laxPrecision) throws IOException {
    FileSystem fs = FileSystem.get(path.toUri(), conf);
    FSDataOutputStream out = fs.create(path);
    try {
      VectorWritable vectorWritable = new VectorWritable(vector);
      vectorWritable.setWritesLaxPrecision(laxPrecision);
      vectorWritable.write(out);
    } finally {
      Closeables.close(out, false);
    }
  }

  public static OpenIntIntHashMap readAsIntMap(Path path, Configuration conf) throws IOException {
    FileSystem fs = FileSystem.get(path.toUri(), conf);
    FSDataInputStream in = fs.open(path);
    try {
      return readAsIntMap(in);
    } finally {
      Closeables.close(in, true);
    }
  }

  /* ugly optimization for loading sparse vectors containing ints only */
  private static OpenIntIntHashMap readAsIntMap(DataInput in) throws IOException {
    int flags = in.readByte();
    Preconditions.checkArgument(flags >> VectorWritable.NUM_FLAGS == 0,
                                "Unknown flags set: %d", Integer.toString(flags, 2));
    boolean dense = (flags & VectorWritable.FLAG_DENSE) != 0;
    boolean sequential = (flags & VectorWritable.FLAG_SEQUENTIAL) != 0;
    boolean laxPrecision = (flags & VectorWritable.FLAG_LAX_PRECISION) != 0;
    Preconditions.checkState(!dense && !sequential, "Only for reading sparse vectors!");

    Varint.readUnsignedVarInt(in);

    OpenIntIntHashMap values = new OpenIntIntHashMap();
    int numNonDefaultElements = Varint.readUnsignedVarInt(in);
    for (int i = 0; i < numNonDefaultElements; i++) {
      int index = Varint.readUnsignedVarInt(in);
      double value = laxPrecision ? in.readFloat() : in.readDouble();
      values.put(index, (int) value);
    }
    return values;
  }

  public static Vector read(Path path, Configuration conf) throws IOException {
    FileSystem fs = FileSystem.get(path.toUri(), conf);
    FSDataInputStream in = fs.open(path);
    try {
      return VectorWritable.readVector(in);
    } finally {
      Closeables.close(in, true);
    }
  }
}
