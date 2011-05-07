package org.apache.mahout.clustering.lda;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;

public class LDADocumentTopicMapper extends Mapper<WritableComparable<?>,VectorWritable,WritableComparable<?>,VectorWritable> {

  private LDAState state;
  private LDAInference infer;

  @Override
  protected void map(WritableComparable<?> key,
                     VectorWritable wordCountsWritable,
                     Context context) throws IOException, InterruptedException {

    Vector wordCounts = wordCountsWritable.get();
    LDAInference.InferredDocument doc;
    try {
      doc = infer.infer(wordCounts);
      context.write(key, new VectorWritable(doc.getGamma().normalize(1)));
    } catch (ArrayIndexOutOfBoundsException e1) {
      throw new IllegalStateException(
         "This is probably because the --numWords argument is set too small.  \n"
         + "\tIt needs to be >= than the number of words (terms actually) in the corpus and can be \n"
         + "\tlarger if some storage inefficiency can be tolerated.", e1);
    }
  }

  public void configure(LDAState myState) {
    this.state = myState;
    this.infer = new LDAInference(state);
  }

  public void configure(Configuration job) {
    try {
      LDAState myState = LDADriver.createState(job);
      configure(myState);
    } catch (IOException e) {
      throw new IllegalStateException("Error creating LDA State!", e);
    }
  }

  @Override
  protected void setup(Context context) {
    configure(context.getConfiguration());
  }
}
