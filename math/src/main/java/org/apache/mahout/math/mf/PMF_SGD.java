package org.apache.mahout.math.mf;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.function.Functions;

/**
 * Created by lmq on 2014/12/24.
 *
 * Probabilistic Matrix Factorization.
 * PMF_SGD: Class for Probabilistic Matrix Factorization using stotastic gradient descent
 * [1] Mnih, Andriy, and Ruslan Salakhutdinov. "Probabilistic matrix factorization." Advances in neural information processing systems. 2007.
 *
 *
 * Probabilistic Matrix Factorization. Factorize a data matrix into two matrices
 * s.t. R = | data - P*Q | = | is minimal.
 * Uses stochastic gradient descent
 *
 */

public class PMF_SGD implements MF {

    /** Arrays for internal storage of V, W and H. */
    private final Matrix R;
    private final Matrix P;
    private final Matrix Q;

    private final double alpha;
    private final double beta1;
    private final double beta2;

    /** Row and column dimensions. */
    private final int n;
    private final int m;

    /** number of features */
    private final int r;

    /** iteration step number before stop*/
    private int stepNum;

    /** value of object function*/
    private double object;

    /** max iteration steps */
    private final int stepMax;



    /** when fabs(oldObjectFunction-newObjectFunction) < errMax, stop iteration*/
    private final double errMax;


    public PMF_SGD(Matrix arg, int r, double alpha, double beta1, double beta2, int stepNum, double errMax) {
        this.n = arg.numRows();
        this.m = arg.numCols();
        this.r = r;
        this.stepNum = 0;
        this.stepMax = stepNum;
        this.errMax = errMax;
        this.alpha = alpha;
        this.beta1 = beta1;
        this.beta2 = beta2;

        this.P = new DenseMatrix(n, r).assign(Functions.random());
        this.Q = new DenseMatrix(r, m).assign(Functions.random());
        this.R = arg;

        this.object = calObject(R, P, Q);
    }

    //@Override
    public void solve() {
        for(; stepNum < stepMax; stepNum++) {
            for(int i=0; i<n; i++) for(int j=0; j<m; j++) {
                if (R.get(i, j) != 0) {
                    double eij = R.get(i, j) - P.viewRow(i).dot(Q.viewColumn(j));
                    for (int k = 0; k < r; k++) {
                        P.set(i, k, P.get(i, k) + alpha * (eij * Q.get(k, j) - beta1 * P.get(i, k)));
                        Q.set(k, j, Q.get(k, j) + alpha * (eij * P.get(i, k) - beta2 * Q.get(k, j)));
                    }
                }
            }

            double newObject = calObject(R, P, Q);

            if(Math.abs(object - newObject) < errMax) {
                object = newObject;
                break;
            }
            object = newObject;
        }

    }

    @Override
    public double calObject(Matrix R, Matrix P, Matrix Q) {
        Matrix PQ = (P.times(Q));
        double err = 0;
        for(int i=0; i<n; i++) for(int j=0; j<m; j++)
            if (R.get(i, j) != 0) err += Math.pow(R.get(i, j) - PQ.get(i, j), 2);
        for(int i=0; i<n; i++) for(int j=0; j<r; j++) err += (beta1) * (Math.pow(P.get(i, j), 2));
        for(int i=0; i<r; i++) for(int j=0; j<m; j++) err += (beta2) * (Math.pow(Q.get(i, j), 2));
        return err/2;
    }

    @Override
    public Matrix getW() {
        return P;
    }

    @Override
    public Matrix getH() {
        return Q;
    }

    @Override
    public int getStep() {
        return stepNum;
    }

    @Override
    public double getObjectFunctionValue(){
        return object;
    }
}
