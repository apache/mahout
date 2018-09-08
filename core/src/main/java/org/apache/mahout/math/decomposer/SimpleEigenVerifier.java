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
package org.apache.mahout.math.decomposer;

import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorIterable;

public class SimpleEigenVerifier implements SingularVectorVerifier {

  @Override
  public EigenStatus verify(VectorIterable corpus, Vector vector) {
    Vector resultantVector = corpus.timesSquared(vector);
    double newNorm = resultantVector.norm(2);
    double oldNorm = vector.norm(2);
    double eigenValue;
    double cosAngle;
    if (newNorm > 0 && oldNorm > 0) {
      eigenValue = newNorm / oldNorm;
      cosAngle = resultantVector.dot(vector) / newNorm * oldNorm;
    } else {
      eigenValue = 1.0;
      cosAngle = 0.0;
    }
    return new EigenStatus(eigenValue, cosAngle, false);
  }

}
