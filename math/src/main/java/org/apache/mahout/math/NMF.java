package org.apache.mahout.math;

/**
 * Created by lmq on 2014/12/19.
 */

/*
* Non-negative Matrix Factorization using Multiplicative update rules
* @author: Liang Mingqiang(2281784755@qq.com)
* Hint: This is the fist time for me to involve in a open source project. And this code is not perfect, Is there anyone willing guide me?
* */




import java.util.Random;
import org.apache.mahout.math.function.DoubleFunction;

public class NMF{

    private final Matrix w;
    private final Matrix h;

    private static int maxInteracoes = 5000;


    /**
     * @param d Matrix original
     * @param r model order
     */
    public NMF(Matrix d, int r) {
        double oldObj, obj;
        int n = d.rowSize();
        int m = d.columnSize();

        w = new DenseMatrix(n, r).assign(RANDOMF);
        h = new DenseMatrix(r, m).assign(RANDOMF);

        for (int i = 0; i < maxInteracoes; i++) {

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

        }

    }

    /**
     * Calcula o valor da função objetivo
     * @param d Matriz original
     * @param w Matriz fator
     * @param h Matriz fator
     * @return Valor da função objetivo
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

}

