package org.apache.mahout.math.mf;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseMatrix;

import java.io.*;

/**
 * Created by lmq on 2014/12/24.
 */
public class MFTestCommon {

    public static Matrix createMatrix(String fileName, int numu, int numi) {
        Matrix R = new SparseMatrix(numu + 2, numi + 2);

        File file = new File(fileName);
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(file));
            String str;
            while ((str = reader.readLine()) != null) {
                String[] ss = str.split("\t");
                R.set(Integer.parseInt(ss[0]),Integer.parseInt(ss[1]), Integer.parseInt(ss[2]));
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                }
            }

            return R;
        }
    }

    public static double calMse(Matrix S, Matrix SS) {
        int cnt = 0;
        double num = 0;
        for(int i=0; i<SS.rowSize(); i++) for(int j=0; j<SS.columnSize(); j++)
            if(SS.get(i, j) != 0) {
                cnt ++;
                num += Math.pow(S.get(i, j)-SS.get(i, j), 2);
            }
        return num/cnt;
    }

    public static double calDensity(Matrix R) {
        double cnt = 0;
        for(int i=0; i<R.rowSize(); i++)
            for(int j=0; j<R.columnSize(); j++)
                if(R.get(i, j) != 0) cnt++;
        return cnt/(R.rowSize()*R.columnSize());
    }

}
