package org.apache.mahout.math.mf;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;


/**
 * Created by lmq on 2014/12/23.
 */
public final class NMF_MUTest {

    @Test
    public void MovielensTest() {
        String trainSet, testSet;
        Matrix R, W, H, RR;
        String resFile = "E:\\Coding\\DataSet\\Result\\NMF_MU_RESULT.txt";
        double result[][][] = new double[10][10][5];

        for(int i=1; i<6; i++) {
            trainSet = String.format("E:\\Coding\\DataSet\\ml-100k\\u%d.base", i);
            testSet  = String.format("E:\\Coding\\DataSet\\ml-100k\\u%d.test", i);
            R  = MFTestCommon.createMatrix(trainSet, 943, 1682);
            RR = MFTestCommon.createMatrix(testSet, 943, 1682);

            NMF_MU nmf;
            for(int k=1; k<6; k++) {
                nmf = new NMF_MU(R, k, 5000, 0.3);
                nmf.solve();
                W = nmf.getW();
                H = nmf.getH();

                result[i][k][0] = k;
                result[i][k][1] = nmf.getStep();
                result[i][k][2] =  MFTestCommon.calMse(W.times(H), RR);
                result[i][k][3] = nmf.getObjectFunctionValue();
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

    }

}
 