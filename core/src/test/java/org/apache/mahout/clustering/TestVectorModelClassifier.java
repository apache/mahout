package org.apache.mahout.clustering;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang.NotImplementedException;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.clustering.canopy.Canopy;
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

public class TestVectorModelClassifier extends MahoutTestCase {

  public void testDMClusterClassification() {
    List<Model<VectorWritable>> models = new ArrayList<Model<VectorWritable>>();
    DistanceMeasure measure = new ManhattanDistanceMeasure();
    models.add(new DistanceMeasureCluster(new DenseVector(2).assign(1), 0, measure));
    models.add(new DistanceMeasureCluster(new DenseVector(2), 1, measure));
    models.add(new DistanceMeasureCluster(new DenseVector(2).assign(-1), 2, measure));
    AbstractVectorClassifier classifier = new VectorModelClassifier(models);
    Vector pdf = classifier.classify(new DenseVector(2));
    assertEquals("[0,0]", "[0.107, 0.787, 0.107]", AbstractCluster.formatVector(pdf, null));
    pdf = classifier.classify(new DenseVector(2).assign(2));
    assertEquals("[2,2]", "[0.867, 0.117, 0.016]", AbstractCluster.formatVector(pdf, null));
  }

  public void testCanopyClassification() {
    List<Model<VectorWritable>> models = new ArrayList<Model<VectorWritable>>();
    DistanceMeasure measure = new ManhattanDistanceMeasure();
    models.add(new Canopy(new DenseVector(2).assign(1), 0, measure));
    models.add(new Canopy(new DenseVector(2), 1, measure));
    models.add(new Canopy(new DenseVector(2).assign(-1), 2, measure));
    AbstractVectorClassifier classifier = new VectorModelClassifier(models);
    Vector pdf = classifier.classify(new DenseVector(2));
    assertEquals("[0,0]", "[0.107, 0.787, 0.107]", AbstractCluster.formatVector(pdf, null));
    pdf = classifier.classify(new DenseVector(2).assign(2));
    assertEquals("[2,2]", "[0.867, 0.117, 0.016]", AbstractCluster.formatVector(pdf, null));
  }

  public void testClusterClassification() {
    List<Model<VectorWritable>> models = new ArrayList<Model<VectorWritable>>();
    DistanceMeasure measure = new ManhattanDistanceMeasure();
    models.add(new Cluster(new DenseVector(2).assign(1), 0, measure));
    models.add(new Cluster(new DenseVector(2), 1, measure));
    models.add(new Cluster(new DenseVector(2).assign(-1), 2, measure));
    AbstractVectorClassifier classifier = new VectorModelClassifier(models);
    Vector pdf = classifier.classify(new DenseVector(2));
    assertEquals("[0,0]", "[0.107, 0.787, 0.107]", AbstractCluster.formatVector(pdf, null));
    pdf = classifier.classify(new DenseVector(2).assign(2));
    assertEquals("[2,2]", "[0.867, 0.117, 0.016]", AbstractCluster.formatVector(pdf, null));
  }

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
    } catch (NotImplementedException e) {
      assertTrue(true);
    }
  }

  public void testSoftClusterClassification() {
    List<Model<VectorWritable>> models = new ArrayList<Model<VectorWritable>>();
    DistanceMeasure measure = new ManhattanDistanceMeasure();
    models.add(new SoftCluster(new DenseVector(2).assign(1), 0, measure));
    models.add(new SoftCluster(new DenseVector(2), 1, measure));
    models.add(new SoftCluster(new DenseVector(2).assign(-1), 2, measure));
    AbstractVectorClassifier classifier = new VectorModelClassifier(models);
    Vector pdf = classifier.classify(new DenseVector(2));
    assertEquals("[0,0]", "[0.000, 1.000, 0.000]", AbstractCluster.formatVector(pdf, null));
    pdf = classifier.classify(new DenseVector(2).assign(2));
    assertEquals("[2,2]", "[0.735, 0.184, 0.082]", AbstractCluster.formatVector(pdf, null));
  }

  public void testGaussianClusterClassification() {
    List<Model<VectorWritable>> models = new ArrayList<Model<VectorWritable>>();
    models.add(new GaussianCluster(new DenseVector(2).assign(1), new DenseVector(2).assign(1), 0));
    models.add(new GaussianCluster(new DenseVector(2), new DenseVector(2).assign(1), 1));
    models.add(new GaussianCluster(new DenseVector(2).assign(-1), new DenseVector(2).assign(1), 2));
    AbstractVectorClassifier classifier = new VectorModelClassifier(models);
    Vector pdf = classifier.classify(new DenseVector(2));
    assertEquals("[0,0]", "[0.107, 0.787, 0.107]", AbstractCluster.formatVector(pdf, null));
    pdf = classifier.classify(new DenseVector(2).assign(2));
    assertEquals("[2,2]", "[0.998, 0.002, 0.000]", AbstractCluster.formatVector(pdf, null));
  }

}
