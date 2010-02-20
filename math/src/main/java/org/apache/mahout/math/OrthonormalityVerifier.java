package org.apache.mahout.math;

import java.util.ArrayList;
import java.util.List;

public class OrthonormalityVerifier {

  public VectorIterable pairwiseInnerProducts(VectorIterable basis) {
    DenseMatrix out = null;
    for(MatrixSlice slice1 : basis) {
      List<Double> dots = new ArrayList<Double>();
      for(MatrixSlice slice2 : basis) {
        dots.add(slice1.vector().dot(slice2.vector()));
      }
      if(out == null) {
        out = new DenseMatrix(dots.size(), dots.size());
      }
      for(int i=0; i<dots.size(); i++) {
        out.set(slice1.index(), i, dots.get(i));
      }
    }
    return out;
  }

}
