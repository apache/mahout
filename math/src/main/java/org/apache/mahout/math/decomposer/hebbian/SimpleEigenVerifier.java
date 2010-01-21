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
package org.apache.mahout.math.decomposer.hebbian;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

public class SimpleEigenVerifier implements SingularVectorVerifier {

  @Override
  public EigenStatus verify(Matrix eigenMatrix, Vector vector) {
    Vector resultantVector = eigenMatrix.timesSquared(vector);
    double newNorm = resultantVector.norm(2);
    double oldNorm = vector.norm(2);
    double eigenValue = (newNorm > 0 && oldNorm > 0) ? newNorm / oldNorm : 1;
    double cosAngle = (newNorm > 0 && oldNorm > 0) ? resultantVector.dot(vector) / (newNorm * oldNorm) : 0;
    return new EigenStatus(eigenValue, cosAngle);
  }

}
