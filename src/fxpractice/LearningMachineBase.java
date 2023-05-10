package fxpractice;

public abstract class LearningMachineBase implements LearningMachine {
    
    /**
     * 特徴量にバイアスを追加します
     * 
     * @param feature   特徴量
     */
    protected double[] addBias(double[] feature) {
        double[] d = new double[feature.length + 1];
        System.arraycopy(feature, 0, d, 0, feature.length);
        d[d.length - 1] = 1;
        return d;
    }
}
