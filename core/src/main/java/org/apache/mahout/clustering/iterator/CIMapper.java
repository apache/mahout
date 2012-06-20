package org.apache.mahout.clustering.iterator;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;

public class CIMapper extends Mapper<WritableComparable<?>,VectorWritable,IntWritable,ClusterWritable> {
  
  private ClusterClassifier classifier;
  private ClusteringPolicy policy;

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();
    String priorClustersPath = conf.get(ClusterIterator.PRIOR_PATH_KEY);
    classifier = new ClusterClassifier();
    classifier.readFromSeqFiles(conf, new Path(priorClustersPath));
    policy = classifier.getPolicy();
    policy.update(classifier);
    super.setup(context);
  }

  @Override
  protected void map(WritableComparable<?> key, VectorWritable value, Context context) throws IOException,
      InterruptedException {
    Vector probabilities = classifier.classify(value.get());
    Vector selections = policy.select(probabilities);
    for (Iterator<Element> it = selections.iterateNonZero(); it.hasNext();) {
      Element el = it.next();
      classifier.train(el.index(), value.get(), el.get());
    }
  }

  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    List<Cluster> clusters = classifier.getModels();
    ClusterWritable cw = new ClusterWritable();
    for (int index = 0; index < clusters.size(); index++) {
      cw.setValue(clusters.get(index));
      context.write(new IntWritable(index), cw);
    }
    super.cleanup(context);
  }
  
}
