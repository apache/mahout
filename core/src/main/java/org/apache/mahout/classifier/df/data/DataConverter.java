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

package org.apache.mahout.classifier.df.data;

import com.google.common.base.Preconditions;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.mahout.math.DenseVector;

import java.util.regex.Pattern;

/**
 * Converts String to Instance using a Dataset
 */
public class DataConverter {

  private static final Pattern COMMA_SPACE = Pattern.compile("[, ]");

  private final Dataset dataset;

  public DataConverter(Dataset dataset) {
    this.dataset = dataset;
  }

  public Instance convert(CharSequence string) {
    // all attributes (categorical, numerical, label), ignored
    int nball = dataset.nbAttributes() + dataset.getIgnored().length;

    String[] tokens = COMMA_SPACE.split(string);
    Preconditions.checkArgument(tokens.length == nball,
        "Wrong number of attributes in the string: " + tokens.length + ". Must be " + nball);

    int nbattrs = dataset.nbAttributes();
    DenseVector vector = new DenseVector(nbattrs);

    int aId = 0;
    for (int attr = 0; attr < nball; attr++) {
      if (!ArrayUtils.contains(dataset.getIgnored(), attr)) {
        String token = tokens[attr].trim();

        if ("?".equals(token)) {
          // missing value
          return null;
        }

        if (dataset.isNumerical(aId)) {
          vector.set(aId++, Double.parseDouble(token));
        } else { // CATEGORICAL
          vector.set(aId, dataset.valueOf(aId, token));
          aId++;
        }
      }
    }

    return new Instance(vector);
  }
}
