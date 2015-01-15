package org.apache.mahout.math.mf;


import org.apache.mahout.math.Matrix;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Created by lmq on 2014/12/24.
 */
public final class MF_SGDTest {

   /* @Test
    public void Convergence_Curve() {
        String trainSet;
        Matrix R;

        trainSet = String.format("E:\\Coding\\DataSet\\ml-100k\\u%d.base", 1);
        R = MFTestCommon.createMatrix(trainSet, 943, 1682);

        MF_SGD mf;
        //mf = new MF_BaseV(R, 2, 0.0002, 500, 0.02);
        mf = new MF_SGD(R, 2, 0.0002, 0.02, 201, 0.02);
        mf.solve();

    }*/

   /* @Test
    public void MovielensTest() {
        String trainSet, testSet;
        Matrix R, W, H, RR;
        String resFile = "E:\\Coding\\DataSet\\Result\\MF_SGD_RESULT.txt";
        double result[][][] = new double[10][10][5];

        for(int i=1; i<6; i++) {
            trainSet = String.format("E:\\Coding\\DataSet\\ml-100k\\u%d.base", i);
            testSet  = String.format("E:\\Coding\\DataSet\\ml-100k\\u%d.test", i);
            R  = MFTestCommon.createMatrix(trainSet, 943, 1682);
            RR = MFTestCommon.createMatrix(testSet, 943, 1682);

            MF_SGD mf;
            for(int k=1; k<6; k++) {
                mf = new MF_SGD(R, k, 0.0002, 0.02, 5000, 0.02);
                mf.solve();
                W = mf.getW();
                H = mf.getH();

                result[i][k][0] = k;
                result[i][k][1] = mf.getStep();
                result[i][k][2] =  MFTestCommon.calMse(W.times(H), RR);
                result[i][k][3] = mf.getObjectFunctionValue();
            }
        }

        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(resFile));
            out.write("Result of MF_SGD on movielens dataset\n");
            for(int i=1; i<6; i++) {
                String str = String.format("Case %d:\nk\tstep\tmse\tobjectfunction\n", i);
                out.write(str);
                for(int j=1; j<6; j++) {
                    str = String.format("%d\t%d\t%f\t%f\n", (int)result[i][j][0], (int)result[i][j][1], result[i][j][2], result[i][j][3]);
                    out.write(str);
                }
            }
            out.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }*/
}
