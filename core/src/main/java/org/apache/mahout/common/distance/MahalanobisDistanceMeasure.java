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

package org.apache.mahout.common.distance;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.parameters.ClassParameter;
import org.apache.mahout.common.parameters.Parameter;
import org.apache.mahout.common.parameters.PathParameter;
import org.apache.mahout.math.Algebra;
import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.SingularValueDecomposition;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.DataInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Collection;
import java.util.List;

//See http://en.wikipedia.org/wiki/Mahalanobis_distance for details
public class MahalanobisDistanceMeasure implements DistanceMeasure {

  private Matrix inverseCovarianceMatrix;
  private Vector meanVector;

  private ClassParameter vectorClass;
  private ClassParameter matrixClass;
  private List<Parameter<?>> parameters;
  private Parameter<Path> inverseCovarianceFile;
  private Parameter<Path> meanVectorFile;

  /*public MahalanobisDistanceMeasure(Vector meanVector,Matrix inputMatrix, boolean inversionNeeded)
  {
    this.meanVector=meanVector;
    if (inversionNeeded)
      setCovarianceMatrix(inputMatrix);
    else
      setInverseCovarianceMatrix(inputMatrix);  
  }*/

  @Override
  public void configure(Configuration jobConf) {
    if (parameters == null) {
      ParameteredGeneralizations.configureParameters(this, jobConf);
    }
    try {
      if (inverseCovarianceFile.get() != null) {
        FileSystem fs = FileSystem.get(inverseCovarianceFile.get().toUri(), jobConf);
        MatrixWritable inverseCovarianceMatrix = 
            ClassUtils.instantiateAs((Class<? extends MatrixWritable>) matrixClass.get(), MatrixWritable.class);
        if (!fs.exists(inverseCovarianceFile.get())) {
          throw new FileNotFoundException(inverseCovarianceFile.get().toString());
        }
        DataInputStream in = fs.open(inverseCovarianceFile.get());
        try {
          inverseCovarianceMatrix.readFields(in);
        } finally {
          Closeables.close(in, true);
        }
        this.inverseCovarianceMatrix = inverseCovarianceMatrix.get();
        Preconditions.checkArgument(this.inverseCovarianceMatrix != null, "inverseCovarianceMatrix not initialized");
      }

      if (meanVectorFile.get() != null) {
        FileSystem fs = FileSystem.get(meanVectorFile.get().toUri(), jobConf);
        VectorWritable meanVector = 
            ClassUtils.instantiateAs((Class<? extends VectorWritable>) vectorClass.get(), VectorWritable.class);
        if (!fs.exists(meanVectorFile.get())) {
          throw new FileNotFoundException(meanVectorFile.get().toString());
        }
        DataInputStream in = fs.open(meanVectorFile.get());
        try {
          meanVector.readFields(in);
        } finally {
          Closeables.close(in, true);
        }
        this.meanVector = meanVector.get();
        Preconditions.checkArgument(this.meanVector != null, "meanVector not initialized");
      }

    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  public Collection<Parameter<?>> getParameters() {
    return parameters;
  }

  @Override
  public void createParameters(String prefix, Configuration jobConf) {
    parameters = Lists.newArrayList();
    inverseCovarianceFile = new PathParameter(prefix, "inverseCovarianceFile", jobConf, null,
            "Path on DFS to a file containing the inverse covariance matrix.");
    parameters.add(inverseCovarianceFile);

    matrixClass = new ClassParameter(prefix, "maxtrixClass", jobConf, DenseMatrix.class,
            "Class<Matix> file specified in parameter inverseCovarianceFile has been serialized with.");
    parameters.add(matrixClass);

    meanVectorFile = new PathParameter(prefix, "meanVectorFile", jobConf, null,
            "Path on DFS to a file containing the mean Vector.");
    parameters.add(meanVectorFile);

    vectorClass = new ClassParameter(prefix, "vectorClass", jobConf, DenseVector.class,
            "Class file specified in parameter meanVectorFile has been serialized with.");
    parameters.add(vectorClass);
  }

  /**
   * @param v The vector to compute the distance to
   * @return Mahalanobis distance of a multivariate vector
   */
  public double distance(Vector v) {
    return Math.sqrt(v.minus(meanVector).dot(Algebra.mult(inverseCovarianceMatrix, v.minus(meanVector))));
  }

  @Override
  public double distance(Vector v1, Vector v2) {
    if (v1.size() != v2.size()) {
      throw new CardinalityException(v1.size(), v2.size());
    }
    return Math.sqrt(v1.minus(v2).dot(Algebra.mult(inverseCovarianceMatrix, v1.minus(v2))));
  }

  @Override
  public double distance(double centroidLengthSquare, Vector centroid, Vector v) {
    return distance(centroid, v); // TODO
  }

  public void setInverseCovarianceMatrix(Matrix inverseCovarianceMatrix) {
    Preconditions.checkArgument(inverseCovarianceMatrix != null, "inverseCovarianceMatrix not initialized");
    this.inverseCovarianceMatrix = inverseCovarianceMatrix;
  }


  /**
   * Computes the inverse covariance from the input covariance matrix given in input.
   *
   * @param m A covariance matrix.
   * @throws IllegalArgumentException if <tt>eigen values equal to 0 found</tt>.
   */
  public void setCovarianceMatrix(Matrix m) {
    if (m.numRows() != m.numCols()) {
      throw new CardinalityException(m.numRows(), m.numCols());
    }
    // See http://www.mlahanas.de/Math/svd.htm for details,
    // which specifically details the case of covariance matrix inversion
    // Complexity: O(min(nm2,mn2))
    SingularValueDecomposition svd = new SingularValueDecomposition(m);
    Matrix sInv = svd.getS();
    // Inverse Diagonal Elems
    for (int i = 0; i < sInv.numRows(); i++) {
      double diagElem = sInv.get(i, i);
      if (diagElem > 0.0) {
        sInv.set(i, i, 1 / diagElem);
      } else {
        throw new IllegalStateException("Eigen Value equals to 0 found.");
      }
    }
    inverseCovarianceMatrix = svd.getU().times(sInv.times(svd.getU().transpose()));
    Preconditions.checkArgument(inverseCovarianceMatrix != null, "inverseCovarianceMatrix not initialized");
  }

  public Matrix getInverseCovarianceMatrix() {
    return inverseCovarianceMatrix;
  }

  public void setMeanVector(Vector meanVector) {
    Preconditions.checkArgument(meanVector != null, "meanVector not initialized");
    this.meanVector = meanVector;
  }

  public Vector getMeanVector() {
    return meanVector;
  }
}
