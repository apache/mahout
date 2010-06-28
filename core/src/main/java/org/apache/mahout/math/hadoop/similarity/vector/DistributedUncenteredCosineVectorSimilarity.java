package org.apache.mahout.math.hadoop.similarity.vector;

import org.apache.mahout.math.hadoop.similarity.Cooccurrence;

/**
 * distributed implementation of cosine similarity that does not center its data
 */
public class DistributedUncenteredCosineVectorSimilarity extends AbstractDistributedVectorSimilarity {

  @Override
  protected double doComputeResult(int rowA, int rowB, Iterable<Cooccurrence> cooccurrences, double weightOfVectorA,
      double weightOfVectorB, int numberOfColumns) {

    int n = 0;
    double sumXY = 0.0;
    double sumX2 = 0.0;
    double sumY2 = 0.0;

    for (Cooccurrence cooccurrence : cooccurrences) {
      double x = cooccurrence.getValueA();
      double y = cooccurrence.getValueB();

      sumXY += x * y;
      sumX2 += x * x;
      sumY2 += y * y;
      n++;
    }

    if (n == 0) {
      return Double.NaN;
    }
    double denominator = Math.sqrt(sumX2) * Math.sqrt(sumY2);
    if (denominator == 0.0) {
      // One or both vectors has -all- the same values;
      // can't really say much similarity under this measure
      return Double.NaN;
    }
    return sumXY / denominator;
  }

}
