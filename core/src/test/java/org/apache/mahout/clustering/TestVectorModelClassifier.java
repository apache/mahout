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

package org.apache.mahout.clustering;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang.NotImplementedException;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.clustering.canopy.Canopy;
import org.apache.mahout.clustering.dirichlet.models.AsymmetricSampledNormalModel;
import org.apache.mahout.clustering.dirichlet.models.GaussianCluster;
import org.apache.mahout.clustering.fuzzykmeans.SoftCluster;
import org.apache.mahout.clustering.kmeans.Cluster;
import org.apache.mahout.clustering.meanshift.MeanShiftCanopy;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Test;

public final class TestVectorModelClassifier extends MahoutTestCase {
  
  @Test
  public void testDMClusterClassification() {
    List<Model<VectorWritable>> models = new ArrayList<Model<VectorWritable>>();
    DistanceMeasure measure = new ManhattanDistanceMeasure();
    models.add(new DistanceMeasureCluster(new DenseVector(2).assign(1), 0,
        measure));
    models.add(new DistanceMeasureCluster(new DenseVector(2), 1, measure));
    models.add(new DistanceMeasureCluster(new DenseVector(2).assign(-1), 2,
        measure));
    AbstractVectorClassifier classifier = new VectorModelClassifier(models);
    Vector pdf = classifier.classify(new DenseVector(2));
    assertEquals("[0,0]", "[0.107, 0.787, 0.107]",
        AbstractCluster.formatVector(pdf, null));
    pdf = classifier.classify(new DenseVector(2).assign(2));
    assertEquals("[2,2]", "[0.867, 0.117, 0.016]",
        AbstractCluster.formatVector(pdf, null));
  }
  
  @Test
  public void testCanopyClassification() {
    List<Model<VectorWritable>> models = new ArrayList<Model<VectorWritable>>();
    DistanceMeasure measure = new ManhattanDistanceMeasure();
    models.add(new Canopy(new DenseVector(2).assign(1), 0, measure));
    models.add(new Canopy(new DenseVector(2), 1, measure));
    models.add(new Canopy(new DenseVector(2).assign(-1), 2, measure));
    AbstractVectorClassifier classifier = new VectorModelClassifier(models);
    Vector pdf = classifier.classify(new DenseVector(2));
    assertEquals("[0,0]", "[0.107, 0.787, 0.107]",
        AbstractCluster.formatVector(pdf, null));
    pdf = classifier.classify(new DenseVector(2).assign(2));
    assertEquals("[2,2]", "[0.867, 0.117, 0.016]",
        AbstractCluster.formatVector(pdf, null));
  }
  
  @Test
  public void testClusterClassification() {
    List<Model<VectorWritable>> models = new ArrayList<Model<VectorWritable>>();
    DistanceMeasure measure = new ManhattanDistanceMeasure();
    models.add(new Cluster(new DenseVector(2).assign(1), 0, measure));
    models.add(new Cluster(new DenseVector(2), 1, measure));
    models.add(new Cluster(new DenseVector(2).assign(-1), 2, measure));
    AbstractVectorClassifier classifier = new VectorModelClassifier(models);
    Vector pdf = classifier.classify(new DenseVector(2));
    assertEquals("[0,0]", "[0.107, 0.787, 0.107]",
        AbstractCluster.formatVector(pdf, null));
    pdf = classifier.classify(new DenseVector(2).assign(2));
    assertEquals("[2,2]", "[0.867, 0.117, 0.016]",
        AbstractCluster.formatVector(pdf, null));
  }
  
  @Test
  public void testMSCanopyClassification() {
    List<Model<VectorWritable>> models = new ArrayList<Model<VectorWritable>>();
    DistanceMeasure measure = new ManhattanDistanceMeasure();
    models.add(new MeanShiftCanopy(new DenseVector(2).assign(1), 0, measure));
    models.add(new MeanShiftCanopy(new DenseVector(2), 1, measure));
    models.add(new MeanShiftCanopy(new DenseVector(2).assign(-1), 2, measure));
    AbstractVectorClassifier classifier = new VectorModelClassifier(models);
    try {
      classifier.classify(new DenseVector(2));
      fail("Expected NotImplementedException");
    } catch (NotImplementedException e) {}
  }
  
  @Test
  public void testSoftClusterClassification() {
    List<Model<VectorWritable>> models = new ArrayList<Model<VectorWritable>>();
    DistanceMeasure measure = new ManhattanDistanceMeasure();
    models.add(new SoftCluster(new DenseVector(2).assign(1), 0, measure));
    models.add(new SoftCluster(new DenseVector(2), 1, measure));
    models.add(new SoftCluster(new DenseVector(2).assign(-1), 2, measure));
    AbstractVectorClassifier classifier = new VectorModelClassifier(models);
    Vector pdf = classifier.classify(new DenseVector(2));
    assertEquals("[0,0]", "[0.000, 1.000, 0.000]",
        AbstractCluster.formatVector(pdf, null));
    pdf = classifier.classify(new DenseVector(2).assign(2));
    assertEquals("[2,2]", "[0.735, 0.184, 0.082]",
        AbstractCluster.formatVector(pdf, null));
  }
  
  @Test
  public void testGaussianClusterClassification() {
    List<Model<VectorWritable>> models = new ArrayList<Model<VectorWritable>>();
    models.add(new GaussianCluster(new DenseVector(2).assign(1),
        new DenseVector(2).assign(1), 0));
    models.add(new GaussianCluster(new DenseVector(2), new DenseVector(2)
        .assign(1), 1));
    models.add(new GaussianCluster(new DenseVector(2).assign(-1),
        new DenseVector(2).assign(1), 2));
    AbstractVectorClassifier classifier = new VectorModelClassifier(models);
    Vector pdf = classifier.classify(new DenseVector(2));
    assertEquals("[0,0]", "[0.212, 0.576, 0.212]",
        AbstractCluster.formatVector(pdf, null));
    pdf = classifier.classify(new DenseVector(2).assign(2));
    assertEquals("[2,2]", "[0.952, 0.047, 0.000]",
        AbstractCluster.formatVector(pdf, null));
  }
  
  @Test
  public void testASNClusterClassification() {
    List<Model<VectorWritable>> models = new ArrayList<Model<VectorWritable>>();
    models.add(new AsymmetricSampledNormalModel(0,
        new DenseVector(2).assign(1), new DenseVector(2).assign(1)));
    models.add(new AsymmetricSampledNormalModel(1, new DenseVector(2),
        new DenseVector(2).assign(1)));
    models.add(new AsymmetricSampledNormalModel(2, new DenseVector(2)
        .assign(-1), new DenseVector(2).assign(1)));
    AbstractVectorClassifier classifier = new VectorModelClassifier(models);
    Vector pdf = classifier.classify(new DenseVector(2));
    assertEquals("[0,0]", "[0.212, 0.576, 0.212]",
        AbstractCluster.formatVector(pdf, null));
    pdf = classifier.classify(new DenseVector(2).assign(2));
    assertEquals("[2,2]", "[0.952, 0.047, 0.000]",
        AbstractCluster.formatVector(pdf, null));
  }
  
}
