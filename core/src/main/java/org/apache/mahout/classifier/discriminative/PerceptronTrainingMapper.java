package org.apache.mahout.classifier.discriminative;

import java.io.IOException;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.Vector;

/**
 * Mapper for perceptron training. Strategy for parallelization:
 * 1) Train separate models on training data samples. Each training
 *    data sample must fit into main memory.
 * 2) Average all trained models into one.
 * */
public class PerceptronTrainingMapper extends
    Mapper<Boolean, Vector, Text, LinearModel> {

  @Override
  protected void map(final Boolean key, final Vector value, Context context) throws IOException, InterruptedException {
    
  }
  
}
