/* Licensed to the Apache Software Foundation (ASF) under one or more
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
import java.util.Collection;
import java.util.List;

import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.clustering.fuzzykmeans.FuzzyKMeansClusterer;
import org.apache.mahout.clustering.fuzzykmeans.SoftCluster;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.TimesFunction;

/**
 * This classifier works with any of the clustering Models. It is initialized with 
 * a list of compatible Models and thereafter it can classify any new Vector into
 * one or more of the Models based upon the pdf() function which each Model supports.
 */
public class VectorModelClassifier extends AbstractVectorClassifier {

  private final List<Model<VectorWritable>> models;

  public VectorModelClassifier(List<Model<VectorWritable>> models) {
    this.models = models;
  }

  @Override
  public Vector classify(Vector instance) {
    Vector pdfs = new DenseVector(models.size());
    if (models.get(0) instanceof SoftCluster) {
      Collection<SoftCluster> clusters = new ArrayList<SoftCluster>();
      List<Double> distances = new ArrayList<Double>();
      for (Model<VectorWritable> model : models) {
        SoftCluster sc = (SoftCluster) model;
        clusters.add(sc);
        distances.add(sc.getMeasure().distance(instance, sc.getCenter()));
      }
      return new FuzzyKMeansClusterer().computePi(clusters, distances);
    } else {
      int i = 0;
      for (Model<VectorWritable> model : models) {
        pdfs.set(i++, model.pdf(new VectorWritable(instance)));
      }
      return pdfs.assign(new TimesFunction(), 1.0 / pdfs.zSum());
    }
  }

  @Override
  public double classifyScalar(Vector instance) {
    if (models.size() == 2) {
      double pdf0 = models.get(0).pdf(new VectorWritable(instance));
      double pdf1 = models.get(1).pdf(new VectorWritable(instance));
      return pdf0 / (pdf0 + pdf1);
    }
    throw new IllegalStateException();
  }

  @Override
  public int numCategories() {
    return models.size();
  }
}
