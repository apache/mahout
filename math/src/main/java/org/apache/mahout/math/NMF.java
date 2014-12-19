package org.apache.mahout.math;

/**
 *
 * * * Created by lmq on 2014/12/19.
 * Non-negative Matrix Factorization using Multiplicative update rules
 * @author: Liang Mingqiang(2281784755@qq.com)
 */




import java.util.Random;
import org.apache.mahout.math.function.DoubleFunction;

public class NMF{
    private Matrix w;
    private Matrix h;


    /**
     * @param d Matrix original
     * @param r model order
     * @param steps max steps before converge
     * @param errMax threshold of object function
     */
    public NMF(Matrix d, int r, int steps, double errMax) {
        double oldObj, obj;
        int n = d.rowSize();
        int m = d.columnSize();

        w = new DenseMatrix(n, r).assign(RANDOMF);
        h = new DenseMatrix(r, m).assign(RANDOMF);

        for (int i = 0; i < steps; i++) {

            oldObj = calObjectFunction(d, w, h);


            Matrix wh = w.times(h);

            Matrix wt = w.transpose();


            //update
            for (int x = 0; x < r; x++) {
                for (int y = 0; y < m; y++) {
                    double value = (wt.times(d).get(x, y))
                            / (wt.times(wh).get(x, y));
                    h.set(x, y, h.get(x, y) * value);
                }
            }


            for (int x = 0; x < n; x++) {
                for (int y = 0; y < r; y++) {
                    Matrix ht = h.transpose();
                    double value = (d.times(ht).get(x, y))
                            / (wh.times(ht).get(x, y));
                    w.set(x, y, w.get(x, y) * value);
                }
            }


            obj = calObjectFunction(d, w, h);
            double erro = oldObj - obj;
            if(erro < errMax) break;

        }

    }

    /**
     * Calculate Value of object function
     * @param d Matrix original
     * @param w Matrix factor
     * @param h Matrix factor
     * @return Value of object function
     */
    static double calObjectFunction(Matrix d, Matrix w, Matrix h){
        Matrix wh = w.times(h);
        Matrix minus = d.minus(wh);
        double err = 0;
        for(int i=0; i<minus.rowSize(); i++)
            for(int j=0; j<minus.columnSize(); j++)
                err += minus.get(i, j) * minus.get(i, j);
        return err;
    }


    private static final DoubleFunction RANDOMF = new DoubleFunction() {
        @Override
        public double apply(double a) {
            Random random = new Random();
            return random.nextDouble();
        }
    };

    public Matrix getW() {
        return w;
    }

    public Matrix getH() {
        return h;
    }

    //unit test
    public static void main(String[] args) {
        Matrix x = new DenseMatrix(3, 3);
        x.viewRow(0).assign(new double[]{1, 2, 3});
        x.viewRow(1).assign(new double[]{2, 4, 6});
        x.viewRow(2).assign(new double[]{3, 6, 9});

        NMF nmf = new NMF(x, 3, 5000, 0.0000002);
        System.out.println(nmf.getH());
        System.out.println(nmf.getW());
        System.out.print(nmf.getH().times(nmf.getW()));
    }

}

