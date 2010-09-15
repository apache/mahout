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

package org.apache.mahout.classifier.sgd;

import com.google.common.collect.Maps;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.QRDecomposition;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectors.Dictionary;

import java.util.Map;
import java.util.Set;

/**
 * Uses sample data to reverse engineer a feature-hashed model.
 *
 * The result gives approximate weights for features and interactions
 * in the original space.
 */
public class ModelDissector {
  int records = 0;
  private Dictionary dict;
  private Matrix a;
  private Matrix b;

  public ModelDissector(int n) {
    a = new SparseRowMatrix(new int[]{Integer.MAX_VALUE, Integer.MAX_VALUE}, true);
    b = new SparseRowMatrix(new int[]{Integer.MAX_VALUE, n});

    dict.intern("Intercept Value");
  }

  public void addExample(Set<String> features, Vector score) {
    for (Vector.Element element : score) {
      b.set(records, element.index(), element.get());
    }

    for (String feature : features) {
      int j = dict.intern(feature);
      a.set(records, j, 1);
    }
    records++;
  }

  public void addExample(Set<String> features, double score) {
    b.set(records, 0, score);

    a.set(records, 0, 1);
    for (String feature : features) {
      int j = dict.intern(feature);
      a.set(records, j, 1);
    }
    records++;
  }

  public Matrix solve() {
    Matrix az = a.viewPart(new int[]{0, 0}, new int[]{records, dict.size()});
    Matrix bz = b.viewPart(new int[]{0, 0}, new int[]{records, b.columnSize()});
    QRDecomposition qr = new QRDecomposition(az.transpose().times(az));
    Matrix x = qr.solve(bz);
    Map<String, Integer> labels = Maps.newHashMap();
    int i = 0;
    for (String s : dict.values()) {
      labels.put(s, i++);
    }
    x.setRowLabelBindings(labels);
    return x;
  }
}
