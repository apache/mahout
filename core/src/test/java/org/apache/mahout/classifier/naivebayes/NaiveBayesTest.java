package org.apache.mahout.classifier.naivebayes;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.classifier.naivebayes.training.TrainNaiveBayesJob;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.MathHelper;
import org.junit.Before;
import org.junit.Test;

import java.io.File;

public class NaiveBayesTest extends MahoutTestCase {

  Configuration conf;
  File inputFile;
  File outputDir;
  File tempDir;

  static final Text LABEL_STOLEN = new Text("stolen");
  static final Text LABEL_NOT_STOLEN = new Text("not_stolen");

  static final Vector.Element COLOR_RED = MathHelper.elem(0, 1);
  static final Vector.Element COLOR_YELLOW = MathHelper.elem(1, 1);
  static final Vector.Element TYPE_SPORTS = MathHelper.elem(2, 1);
  static final Vector.Element TYPE_SUV = MathHelper.elem(3, 1);
  static final Vector.Element ORIGIN_DOMESTIC = MathHelper.elem(4, 1);
  static final Vector.Element ORIGIN_IMPORTED = MathHelper.elem(5, 1);


  @Before
  public void setup() throws Exception {
    super.setUp();

    conf = new Configuration();

    inputFile = getTestTempFile("trainingInstances.seq");
    outputDir = getTestTempDir("output");
    outputDir.delete();
    tempDir = getTestTempDir("tmp");

    SequenceFile.Writer writer = new SequenceFile.Writer(FileSystem.get(conf), conf,
        new Path(inputFile.getAbsolutePath()), Text.class, VectorWritable.class);

    try {
      writer.append(LABEL_STOLEN,      trainingInstance(COLOR_RED, TYPE_SPORTS, ORIGIN_DOMESTIC));
      writer.append(LABEL_NOT_STOLEN, trainingInstance(COLOR_RED, TYPE_SPORTS, ORIGIN_DOMESTIC));
      writer.append(LABEL_STOLEN,      trainingInstance(COLOR_RED, TYPE_SPORTS, ORIGIN_DOMESTIC));
      writer.append(LABEL_NOT_STOLEN, trainingInstance(COLOR_YELLOW, TYPE_SPORTS, ORIGIN_DOMESTIC));
      writer.append(LABEL_STOLEN,      trainingInstance(COLOR_YELLOW, TYPE_SPORTS, ORIGIN_IMPORTED));
      writer.append(LABEL_NOT_STOLEN, trainingInstance(COLOR_YELLOW, TYPE_SUV, ORIGIN_IMPORTED));
      writer.append(LABEL_STOLEN,      trainingInstance(COLOR_YELLOW, TYPE_SUV, ORIGIN_IMPORTED));
      writer.append(LABEL_NOT_STOLEN, trainingInstance(COLOR_YELLOW, TYPE_SUV, ORIGIN_DOMESTIC));
      writer.append(LABEL_NOT_STOLEN, trainingInstance(COLOR_RED, TYPE_SUV, ORIGIN_IMPORTED));
      writer.append(LABEL_STOLEN,      trainingInstance(COLOR_RED, TYPE_SPORTS, ORIGIN_IMPORTED));
    } finally {
      Closeables.closeQuietly(writer);
    }
  }

  @Test
  public void toyData() throws Exception {
    TrainNaiveBayesJob trainNaiveBayes = new TrainNaiveBayesJob();
    trainNaiveBayes.setConf(conf);
    trainNaiveBayes.run(new String[] { "--input", inputFile.getAbsolutePath(), "--output", outputDir.getAbsolutePath(),
        "--labels", "stolen,not_stolen", "--tempDir", tempDir.getAbsolutePath() });

    NaiveBayesModel naiveBayesModel = NaiveBayesModel.materialize(new Path(outputDir.getAbsolutePath()), conf);

    AbstractVectorClassifier classifier = new StandardNaiveBayesClassifier(naiveBayesModel);

    assertEquals(2, classifier.numCategories());

    Vector prediction = classifier.classify(trainingInstance(COLOR_RED, TYPE_SUV, ORIGIN_DOMESTIC).get());

    // should be classified as not stolen
    assertTrue(prediction.get(0) < prediction.get(1));
  }

  @Test
  public void toyDataComplementary() throws Exception {
    TrainNaiveBayesJob trainNaiveBayes = new TrainNaiveBayesJob();
    trainNaiveBayes.setConf(conf);
    trainNaiveBayes.run(new String[] { "--input", inputFile.getAbsolutePath(), "--output", outputDir.getAbsolutePath(),
        "--labels", "stolen,not_stolen", "--trainComplementary", String.valueOf(true),
        "--tempDir", tempDir.getAbsolutePath() });

    NaiveBayesModel naiveBayesModel = NaiveBayesModel.materialize(new Path(outputDir.getAbsolutePath()), conf);

    AbstractVectorClassifier classifier = new ComplementaryNaiveBayesClassifier(naiveBayesModel);

    assertEquals(2, classifier.numCategories());

    Vector prediction = classifier.classify(trainingInstance(COLOR_RED, TYPE_SUV, ORIGIN_DOMESTIC).get());

    // should be classified as not stolen
    assertTrue(prediction.get(0) < prediction.get(1));
  }

  VectorWritable trainingInstance(Vector.Element... elems) {
    DenseVector trainingInstance = new DenseVector(6);
    for (Vector.Element elem : elems) {
      trainingInstance.set(elem.index(), elem.get());
    }
    return new VectorWritable(trainingInstance);
  }


}
