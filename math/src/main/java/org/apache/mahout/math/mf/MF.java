package org.apache.mahout.math.mf;

/**
 * Created by lmq on 2014/12/23.
 */

import org.apache.mahout.math.Matrix;

/**
 * Matrix factorization: to predict the null value in a sparse matrix
 * */

public interface MF {
    Matrix getW();

    Matrix getH();

    void solve();

    int getStep();

    double calObject(Matrix V, Matrix W, Matrix H);

    double getObjectFunctionValue();
}
