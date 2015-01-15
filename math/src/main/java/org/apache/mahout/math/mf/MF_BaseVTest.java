package org.apache.mahout.math.mf;


import org.apache.mahout.math.Matrix;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Created by lmq on 2014/12/24.
 */
public final class MF_BaseVTest {

    /*@Test
    public void Convergence_Curve() {
        String trainSet;
        Matrix R;

        trainSet = String.format("E:\\Coding\\DataSet\\ml-100k\\u%d.base", 1);
        R = MFTestCommon.createMatrix(trainSet, 943, 1682);

        MF_BaseV mf;
        mf = new MF_BaseV(R, 2, 0.0002, 201, 0.02);
        mf.solve();

    }*/


    @Test
    public void CVMse() {
        String trainSet, testSet;
        Matrix R, W, H, RR;
        String resFile = "E:\\Coding\\DataSet\\Result\\MF_Base_RESULT.txt";
        double result[][] = new double[10][10];

        for(int i=1; i<6; i++) {
            trainSet = String.format("E:\\Coding\\DataSet\\ml-100k\\u%d.base", i);
            testSet = String.format("E:\\Coding\\DataSet\\ml-100k\\u%d.test", i);
            R = MFTestCommon.createMatrix(trainSet, 943, 1682);
            RR = MFTestCommon.createMatrix(testSet, 943, 1682);

            MF_BaseV mf;

            for(int k=1; k<6; k++) {

                mf = new MF_BaseV(R, k, 0.0002, 5000, 0.02);
                mf.solve();
                W = mf.getW();
                H = mf.getH();

                result[k][0] = k; //rank
                result[k][1] += mf.getStep(); //step
                result[k][2] += MFTestCommon.calMse(W.times(H), RR); //mse
                result[k][3] += mf.getObjectFunctionValue(); //objectfunction
                result[k][4] += MFTestCommon.calDensity(R);
                result[k][5] += MFTestCommon.calDensity(W.times(H));
            }

        }


        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(resFile));
            out.write("Result of MF_Base on movielens dataset\n");
            for(int i=1; i<6; i++) {
                String str = String.format("k\tstep\tmse\tobjectfunction\tDensity_Before\tDensity_After\n");
                out.write(str);
                str = String.format("%d\t%d\t%f\t%f\t%f\t%f\n",
                        (int)result[i][0], (int)result[i][1]/5, result[i][2]/5, result[i][3]/5, result[i][4]/5, result[i][5]/5);
                out.write(str);
            }
            out.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
