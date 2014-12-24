package org.apache.mahout.math.mf;

/**
 * Created by lmq on 2014/12/23.
 */

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.function.Functions;


/**
 * implement an in-memory Non-negative Matrix Factorization using multiplicative update rules
 * */

public class NMF_MU implements MF {

    /** Arrays for internal storage of V, W and H. */
    private final Matrix V;
    private final Matrix W;
    private final Matrix H;

    /** Row and column dimensions. */
    private final int n;
    private final int m;

    /** number of features */
    private final int k;

    /** max iteration steps */
    private final int stepMax;

    /** iteration step number before stop*/
    private int stepNum;

    /** value of object function*/
    private double object;

    /** when fabs(oldObjectFunction-newObjectFunction) < errMax, stop iteration*/
    private final double errMax;

    public NMF_MU(Matrix arg, int k, int stepNum, double errMax) {
        n = arg.numRows();
        m = arg.numCols();
        this.k = k;
        this.stepNum = 0;
        this.stepMax = stepNum;
        this.errMax = errMax;

        W = new DenseMatrix(n, k).assign(Functions.random());
        H = new DenseMatrix(k, m).assign(Functions.random());
        V = arg;

        this.object = calObject(V,  W, H);
    }

    @Override
    public void solve() {
        /** parameter epsilon is used to avoid division by zero*/
        final double epsilon = 1.0E-9;

        for(; stepNum < stepMax; stepNum++) {
            //update H
            Matrix WtV = W.transpose().times(V);
            Matrix WtWH = W.transpose().times(W.times(H));
            for(int i=0; i<k; i++) for(int j=0; j<m; j++)
                H.set(i, j, H.get(i, j) * (WtV.get(i, j) / (WtWH.get(i, j)+epsilon)));

            //update W
            Matrix VHt = V.times(H.transpose());
            Matrix WHHt = W.times(H).times(H.transpose());
            for(int i=0; i<n; i++) for(int j=0; j<k; j++)
                W.set(i, j, W.get(i, j) * (VHt.get(i, j) / (WHHt.get(i, j)+epsilon)));

            double newObject = calObject(V, W, H);
            System.out.printf("%d %f %f\n", stepNum, newObject, Math.abs(object-newObject));
            if(Math.abs(object - newObject) < errMax) {
                object = newObject;
                break;
            }
            object = newObject;
        }
    }

    @Override
    public double calObject(Matrix V, Matrix W, Matrix H) {
        Matrix WH = (W.times(H));
        double err = 0;
        for(int i=0; i<n; i++) for(int j=0; j<m; j++)
            if(V.get(i, j) != 0) err += Math.pow(V.get(i, j)-WH.get(i, j), 2);
        return err;
    }

    @Override
    public Matrix getW() {
        return W;
    }

    @Override
    public Matrix getH() {
        return H;
    }

    @Override
    public int getStep() {
        return stepNum;
    }

    @Override
    public double getObjectFunctionValue() {
        return object;
    }
}
