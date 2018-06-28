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

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.ConstantValueEncoder;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.encoders.StaticWordValueEncoder;

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;

public class TelephoneCall {
  public static final int FEATURES = 100;
  private static final ConstantValueEncoder interceptEncoder = new ConstantValueEncoder("intercept");
  private static final FeatureVectorEncoder featureEncoder = new StaticWordValueEncoder("feature");

  private RandomAccessSparseVector vector;

  private Map<String, String> fields = new LinkedHashMap<>();

  public TelephoneCall(Iterable<String> fieldNames, Iterable<String> values) {
    vector = new RandomAccessSparseVector(FEATURES);
    Iterator<String> value = values.iterator();
    interceptEncoder.addToVector("1", vector);
    for (String name : fieldNames) {
      String fieldValue = value.next();
      fields.put(name, fieldValue);

      switch (name) {
        case "age": {
          double v = Double.parseDouble(fieldValue);
          featureEncoder.addToVector(name, Math.log(v), vector);
          break;
        }
        case "balance": {
          double v;
          v = Double.parseDouble(fieldValue);
          if (v < -2000) {
            v = -2000;
          }
          featureEncoder.addToVector(name, Math.log(v + 2001) - 8, vector);
          break;
        }
        case "duration": {
          double v;
          v = Double.parseDouble(fieldValue);
          featureEncoder.addToVector(name, Math.log(v + 1) - 5, vector);
          break;
        }
        case "pdays": {
          double v;
          v = Double.parseDouble(fieldValue);
          featureEncoder.addToVector(name, Math.log(v + 2), vector);
          break;
        }
        case "job":
        case "marital":
        case "education":
        case "default":
        case "housing":
        case "loan":
        case "contact":
        case "campaign":
        case "previous":
        case "poutcome":
          featureEncoder.addToVector(name + ":" + fieldValue, 1, vector);
          break;
        case "day":
        case "month":
        case "y":
          // ignore these for vectorizing
          break;
        default:
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
