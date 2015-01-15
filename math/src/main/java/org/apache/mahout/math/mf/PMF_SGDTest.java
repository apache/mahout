package org.apache.mahout.math.mf;

import org.apache.mahout.math.Matrix;
import org.junit.Test;

/**
 * Created by lmq on 2015/1/15.
 */
public class PMF_SGDTest {

   /* @Test
    public void Convergence_Curve() {
        String trainSet;
        Matrix R;

        trainSet = String.format("E:\\Coding\\DataSet\\ml-100k\\u%d.base", 1);
        R = MFTestCommon.createMatrix(trainSet, 943, 1682);

        PMF_SGD mf;
        //mf = new MF_BaseV(R, 2, 0.0002, 500, 0.02);
        mf = new PMF_SGD(R, 2, 0.0002, 0.02, 0.02, 201, 0.02);
        mf.solve();

    }*/

}
