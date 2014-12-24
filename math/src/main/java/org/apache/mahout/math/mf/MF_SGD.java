package org.apache.mahout.math.mf;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.function.Functions;

/**
 * Created by lmq on 2014/12/24.
 */
public class MF_SGD implements MF{

    /** Arrays for internal storage of V, W and H. */
    private final Matrix V;
    private final Matrix W;
    private final Matrix H;

    private final double alpha;
    private final double beta;

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


    public MF_SGD(Matrix arg, int r, double alpha, double beta, int stepNum, double errMax) {
        this.n = arg.numRows();
        this.m = arg.numCols();
        this.r = r;
        this.stepNum = 0;
        this.stepMax = stepNum;
        this.errMax = errMax;
        this.alpha = alpha;
        this.beta = beta;

        this.W = new DenseMatrix(n, r).assign(Functions.random());
        this.H = new DenseMatrix(r, m).assign(Functions.random());
        this.V = arg;

        this.object = calObject(V,  W, H);
    }

    //@Override
    public void solve() {
        for(; stepNum < stepMax; stepNum++) {
            for(int i=0; i<n; i++) for(int j=0; j<m; j++)
                if(V.get(i, j) != 0) {
                    double eij = V.get(i, j)-W.viewRow(i).dot(H.viewColumn(j));
                    for(int k=0; k<r; k++) {
                        W.set(i, k, W.get(i, k)+alpha*(2*eij*H.get(k, j)-beta*W.get(i, k)));
                        H.set(k, j, H.get(k, j)+alpha*(2*eij*W.get(i, k)-beta*H.get(k, j)));
                    }
                }

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
            if(V.get(i, j) != 0) {
                err += Math.pow(V.get(i, j)-WH.get(i, j), 2);
                for(int k=0; k<r; k++) {
                    err += (beta/2)*(Math.pow(W.get(i,k), 2)+Math.pow(H.get(k,j), 2));
                }
            }
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
    public double getObjectFunctionValue(){
        return object;
    }
}
