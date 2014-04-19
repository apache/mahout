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

package org.apache.mahout.classifier.sgd.bankmarketing;

import com.google.common.collect.Maps;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.ConstantValueEncoder;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.encoders.StaticWordValueEncoder;

import java.util.Iterator;
import java.util.Map;

public class TelephoneCall {
  public static final int FEATURES = 100;
  private static final ConstantValueEncoder interceptEncoder = new ConstantValueEncoder("intercept");
  private static final FeatureVectorEncoder featureEncoder = new StaticWordValueEncoder("feature");

  private RandomAccessSparseVector vector;

  private Map<String, String> fields = Maps.newLinkedHashMap();

  public TelephoneCall(Iterable<String> fieldNames, Iterable<String> values) {
    vector = new RandomAccessSparseVector(FEATURES);
    Iterator<String> value = values.iterator();
    interceptEncoder.addToVector("1", vector);
    for (String name : fieldNames) {
      String fieldValue = value.next();
      fields.put(name, fieldValue);

      if (name.equals("age")) {
        double v = Double.parseDouble(fieldValue);
        featureEncoder.addToVector(name, Math.log(v), vector);

      } else if (name.equals("balance")) {
        double v;
        v = Double.parseDouble(fieldValue);
        if (v < -2000) {
          v = -2000;
        }
        featureEncoder.addToVector(name, Math.log(v + 2001) - 8, vector);

      } else if (name.equals("duration")) {
        double v;
        v = Double.parseDouble(fieldValue);
        featureEncoder.addToVector(name, Math.log(v + 1) - 5, vector);

      } else if (name.equals("pdays")) {
        double v;
        v = Double.parseDouble(fieldValue);
        featureEncoder.addToVector(name, Math.log(v + 2), vector);

      } else if (name.equals("job") || name.equals("marital") || name.equals("education") || name.equals("default") ||
                 name.equals("housing") || name.equals("loan") || name.equals("contact") || name.equals("campaign") ||
                 name.equals("previous") || name.equals("poutcome")) {
        featureEncoder.addToVector(name + ":" + fieldValue, 1, vector);

      } else if (name.equals("day") || name.equals("month") || name.equals("y")) {
        // ignore these for vectorizing
      } else {
        throw new IllegalArgumentException(String.format("Bad field name: %s", name));
      }
    }
  }

  public Vector asVector() {
    return vector;
  }

  public int getTarget() {
    return fields.get("y").equals("no") ? 0 : 1;
  }
}
