package org.apache.mahout.math.mf;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.function.Functions;

<<<<<<< HEAD
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Created by lmq on 2014/12/27.
 */
public class PMF_SGD implements MF {

    /** Arrays for internal storage of V, W and H. */
    private final Matrix V;
    private final Matrix W;
    private final Matrix H;

    private final double alpha;
    private final double lambdap;
    private final double lambdaq;
=======
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
>>>>>>> 324ac412b16823924f660db04425309d097068c3

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


<<<<<<< HEAD
    public PMF_SGD(Matrix arg, int r, double alpha, double lambdap, double lambdaq, int stepNum, double errMax) {
=======
    public PMF_SGD(Matrix arg, int r, double alpha, double beta1, double beta2, int stepNum, double errMax) {
>>>>>>> 324ac412b16823924f660db04425309d097068c3
        this.n = arg.numRows();
        this.m = arg.numCols();
        this.r = r;
        this.stepNum = 0;
        this.stepMax = stepNum;
        this.errMax = errMax;
        this.alpha = alpha;
<<<<<<< HEAD
        this.lambdap = lambdap;
        this.lambdaq = lambdaq;

        this.W = new DenseMatrix(n, r).assign(Functions.random());
        this.H = new DenseMatrix(r, m).assign(Functions.random());
        this.V = arg;

        this.object = calObject(V,  W, H);

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
    public void solve() {

      /*  String resFile = "E:\\Coding\\DataSet\\Result\\ConvergenceCurve\\PMF_ConvergenceCurve.txt";
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(resFile));*/


        for(; stepNum < stepMax; stepNum++) {
            for(int i=0; i<n; i++) for(int j=0; j<m; j++)
                if(V.get(i, j) != 0) {
                    double eij = V.get(i, j)-W.viewRow(i).dot(H.viewColumn(j));
                    for(int k=0; k<r; k++) {
                        W.set(i, k, W.get(i, k)+alpha*(eij*H.get(k, j)-lambdap*W.get(i, k)));
                        H.set(k, j, H.get(k, j)+alpha*(eij*W.get(i, k)-lambdaq*H.get(k, j)));
                    }
                }

            double newObject = calObject(V, W, H);
            System.out.printf("%d %f %f\n", stepNum, newObject, Math.abs(object-newObject));

            /*String str = String.format("%d\t %f\n", stepNum, newObject);
            out.write(str);
            out.flush();*/

=======
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
>>>>>>> 324ac412b16823924f660db04425309d097068c3

            if(Math.abs(object - newObject) < errMax) {
                object = newObject;
                break;
            }
            object = newObject;
        }

<<<<<<< HEAD

       /* } catch (IOException e) {
            e.printStackTrace();
        }*/


=======
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
>>>>>>> 324ac412b16823924f660db04425309d097068c3
    }

    @Override
    public int getStep() {
        return stepNum;
    }

<<<<<<< HEAD
    //@Override
    private double calObject(Matrix V, Matrix W, Matrix H) {
        Matrix WH = (W.times(H));
        double err = 0;
        for(int i=0; i<n; i++) for(int j=0; j<m; j++)
            if(V.get(i, j) != 0) err += Math.pow(V.get(i, j)-WH.get(i, j), 2);
        for(int i=0; i<W.rowSize(); i++) for(int j=0; j<W.columnSize(); j++)
            err += lambdap*Math.pow(W.get(i,j), 2) + lambdaq*Math.pow(H.get(j,i), 2);

        return err/2;
    }

    @Override
    public double getObjectFunctionValue() {
=======
    @Override
    public double getObjectFunctionValue(){
>>>>>>> 324ac412b16823924f660db04425309d097068c3
        return object;
    }
}
