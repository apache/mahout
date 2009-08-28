package org.apache.mahout.utils;
import java.beans.ConstructorProperties;
import java.io.Serializable;

public class TimingStatistics implements Serializable {
    private int nCalls;
    private long minTime;
    private long maxTime;
    private long sumTime;
    private double sumSquaredTime;
    
    /** Creates a new instance of CallStats */
    public TimingStatistics() {
    }
    
    @ConstructorProperties({"nCalls", "minTime", "maxTime", "sumTime",
                            "sumSquaredTime"})
    public TimingStatistics(int nCalls, long minTime, long maxTime, long sumTime,
                     double sumSquaredTime) {
        this.nCalls = nCalls;
        this.minTime = minTime;
        this.maxTime = maxTime;
        this.sumTime = sumTime;
        this.sumSquaredTime = sumSquaredTime;
    }
    
    public int getNCalls() {
        return nCalls;
    }
    
    public long getMinTime() {
        return Math.max(0, minTime);
    }
    
    public long getMaxTime() {
        return maxTime;
    }
    
    public long getSumTime() {
        return sumTime;
    }
    
    public double getSumSquaredTime() {
        return sumSquaredTime;
    }
    
    public long getMeanTime() {
        if (nCalls == 0)
            return 0;
        else
            return sumTime / nCalls;
    }
    
    public long getStdDevTime() {
        if (nCalls == 0)
            return 0;
        double mean = getMeanTime();
        double meanSquared = mean * mean;
        double meanOfSquares = sumSquaredTime / nCalls;
        double variance = meanOfSquares - meanSquared;
        if (variance < 0)
            return 0;  // might happen due to rounding error
        return (long) Math.sqrt(variance);
    }
    
    public String toString() {
        return "\n" +
        		"nCalls = " + nCalls + ";\n" +
                "sumTime = " + getSumTime()/1000000000f + "s;\n" +
                "minTime = " + minTime/1000000f + "ms;\n" +
                "maxTime = " + maxTime/1000000f + "ms;\n" +
                "meanTime = " + getMeanTime()/1000000f + "ms;\n" +
                "stdDevTime = " + getStdDevTime()/1000000f + "ms;";
    }
    
    public Call newCall() {
        return new Call();
    }
    
    public class Call {
        private final long startTime = System.nanoTime();
        private Call() {}
        
        public void end() {
            long elapsed = System.nanoTime() - startTime;
            synchronized (TimingStatistics.this) {
                nCalls++;
                if (elapsed < minTime || nCalls == 1)
                    minTime = elapsed;
                if (elapsed > maxTime)
                    maxTime = elapsed;
                sumTime += elapsed;
                double elapsedFP = elapsed;
                sumSquaredTime += elapsedFP * elapsedFP;
            }
        }
    }
}
