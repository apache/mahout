/**
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

package org.apache.mahout.df.data;

import java.util.StringTokenizer;

import org.apache.commons.lang.ArrayUtils;
import org.apache.mahout.matrix.DenseVector;

/**
 * Converts String to Instance using a Dataset
 */
public class DataConverter {

  private final Dataset dataset;

  public DataConverter(Dataset dataset) {
    this.dataset = dataset;
  }

  public Instance convert(int id, String string) {
    // all attributes (categorical, numerical), ignored, label
    int nball = dataset.nbAttributes() + dataset.getIgnored().length + 1;

    StringTokenizer tokenizer = new StringTokenizer(string, ", ");
    if (tokenizer.countTokens() != nball) {
      throw new IllegalArgumentException("Wrong number of attributes in the string");
    }

    int nbattrs = dataset.nbAttributes();
    DenseVector vector = new DenseVector(nbattrs);

    int aId = 0;
    int label = -1;
    for (int attr = 0; attr < nball; attr++) {
      String token = tokenizer.nextToken();
      
      if (ArrayUtils.contains(dataset.getIgnored(), attr)) {
        continue; // IGNORED
      }
      
      if (attr == dataset.getLabelId()) {
        label = dataset.labelCode(token);
      } else if (dataset.isNumerical(aId)) {
        vector.set(aId++, Double.parseDouble(token));
      } else {
        vector.set(aId, dataset.valueOf(aId, token));
        aId++;
      }
    }

    if (label == -1)
      throw new IllegalStateException("Label not found!");

    return new Instance(id, vector, label);
  }
}
