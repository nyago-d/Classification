package fxpractice;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

/**
 * さぽーとべくたーましーん（強）
 */
public class SVM2 implements LearningMachine {
    
    /** 認識したパターン */
    private final List<LearningData> learning = new ArrayList<>();
    
    /** 学習係数 */
    private final double learningRate = 0.2;
    
    /** 最大更新回数 */
    private final double maxIteration = 5000;
    
    /** 重みベクトル（配列のサイズは入力ベクトルの次元） */
    private double[] weight;
    
    /** バイアス */
    private double bias;
    
    /**
     * コンストラクタ
     */
    public SVM2(int futureSize) {
        this.weight = new double[futureSize];
    }
    
    @Override
    public void add(int cls, double[] data) {
        this.learning.add(new LearningData(cls, data));
    }
    
    @Override
    public void learn() {
        
        // 特徴量スケーリングする
        this.learning.forEach(ld -> ld.scalingFeature = this.scaling(ld.feature));
        
        // 正負各側のサポートベクターが確定したか
        boolean isFixPositive = false;
        boolean isFixNegative = false;
        
        // 正負のデータに分ける
        Map<Boolean, List<LearningData>> partition = this.learning.stream()
                                                                  .collect(Collectors.partitioningBy(ld -> ld.lavel >= 0));
        List<LearningData> positive = partition.get(true);
        List<LearningData> negative = partition.get(false);
        
        // 勾配法で未定乗数を求める
        // λ>=0の条件とsum(λ*label)=0の条件はどう満たせばいいの？
        for (int i = 0; i < maxIteration; i++) {

            if (!isFixPositive) {
                
                boolean isFilter = false;
                
                double[] positiveDelta = positive.stream().mapToDouble(ld -> this.learningRate * this.lagrange_d(ld)).toArray();
                
                for (int j = 0; j < positive.size(); j++) {
                    LearningData ld = positive.get(j);
                    ld.lambda += positiveDelta[j];
                    if (ld.lambda < 0) {
                        ld.lambda = 0;
                        isFilter = true;
                    }
                }
                
                if (isFilter) {
                    
                    positive = positive.stream().filter(ld -> ld.lambda > 0).collect(Collectors.toList());
                    
                    if (positive.stream().filter(ld -> ld.isSupportVector()).count() == 1) {
                        isFixPositive = true;
                    } else {
                        double positiveSum = positive.stream().mapToDouble(ld -> ld.lambda).sum();
                        positive.stream().forEach(ld -> ld.lambda = ld.lambda / positiveSum);
                    }
                }
            }
            
            if (!isFixNegative) {
                
                boolean isFilter = false;
                
                double[] negativeDelta = negative.stream().mapToDouble(ld -> this.learningRate * this.lagrange_d(ld)).toArray();
                
                for (int j = 0; j < negative.size(); j++) {
                    LearningData ld = negative.get(j);
                    ld.lambda += negativeDelta[j];
                    if (ld.lambda < 0) {
                        ld.lambda = 0;
                        isFilter = true;
                    }
                }
                
                if (isFilter) {
                    
                    negative = negative.stream().filter(ld -> ld.lambda > 0).collect(Collectors.toList());
                    
                    if (negative.stream().filter(ld -> ld.isSupportVector()).count() == 1) {
                        isFixNegative = true;
                    } else {
                        double positiveSum = negative.stream().mapToDouble(ld -> ld.lambda).sum();
                        negative.stream().forEach(ld -> ld.lambda = ld.lambda / positiveSum);
                    }
                }
            }
            
            if (isFixPositive && isFixNegative) {
                break;
            }
        }
        
        // ラグランジュ乗数を確認
        System.out.println(this.learning.stream().map(ld -> String.valueOf(ld.lambda)).collect(Collectors.joining(" ")));

        // サポートベクトルを抜き出し
        List<LearningData> supportVectors = this.learning.stream().filter(ld -> ld.isSupportVector()).collect(Collectors.toList());

        // 重みベクトルを更新
        for (int i = 0; i < this.weight.length; i++) {
            final int idx = i;
            this.weight[i] = supportVectors.stream()
                                           .mapToDouble(ld -> ld.lambda * ld.lavel * ld.scalingFeature[idx])
                                           .sum();
        }
        
        // バイアスを更新（b = (1 / N_s)・∑(t_n - ∑λ_m・t_m・k(xn・xm)）
        double sum = 0.;
        for (int n = 0; n < supportVectors.size(); n++) {
            LearningData nn = supportVectors.get(n);
            double tmp = 0.;
            for (int m = 0; m < supportVectors.size(); m++) {
                LearningData mm = supportVectors.get(m);
                tmp += mm.lambda * mm.lavel * this.kernel(nn.scalingFeature, mm.scalingFeature);
            }
            sum += nn.lavel - tmp;
        }
        this.bias = sum / supportVectors.size();
    }
    
    /**
     * ラグランジュ関数の微分
     * 
     * ∂L/∂a_n = 1 - ∑(λ・tn・tm・k(xn・xm))
     */
    private double lagrange_d(LearningData learningData) {
        
        double sum = this.learning.stream().filter(ld -> ld.lambda > 0)
                                           .mapToDouble(ld -> ld.lambda * learningData.lavel * ld.lavel * this.kernel(learningData.scalingFeature, ld.scalingFeature))
                                           .sum();
        
        return 1 - sum;
    } 
    
    /**
     * カーネル
     */
    private double kernel(double[] x, double[] y) {
        return this.dot(x, y);  // 線形カーネル
    }
    
    /**
     * 内積
     */
    private double dot(double[] x, double[] y) {
        double res = 0;
        for (int i = 0; i < x.length; i++) {
            res += x[i] * y[i];
        }
        return res;
    }
    
    /**
     * 入力データをスケーリングする
     * 
     * TODO スケールの幅をどう決めるか
     * 
     * @param data
     * @return 
     */
    private double[] scaling(double[] data) {
        double[] res = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            res[i] = data[i] / 200 - 1;
        }
        return res;
    }

    @Override
    public int predict(double[] data) {
        return this.dot(this.scaling(data), weight) + bias > 0 ? 1 : -1;
    }

    @Override
    public void draw(GraphicsContext gc) {
        
        gc.setStroke(Color.GREEN);
        learning.stream().filter(ld -> ld.isSupportVector())
                .forEach(ld -> gc.strokeOval(ld.feature[0] - 2, ld.feature[1] - 2, 9, 9));
        
        int w = (int) gc.getCanvas().getWidth();
        int h = (int) gc.getCanvas().getHeight();
        
        for (int x = 0; x < w; x += 2) {
            for (int y = 0; y < h; y += 2) {
                int ans = this.predict(new double[]{x, y});
                gc.setFill(ans > 0 ? Color.BLUE : Color.RED);
                gc.fillOval(x, y, 1, 1);
            }
        }
    }

    @Override
    public void reset() {
        this.learning.clear();
        this.weight = new double[this.weight.length];
    }

    @Override
    public String getTitle() {
        return "サポートベクターマシン（ハードマージン）";
    }
    
    /**
     * 学習データ
     */
    private static class LearningData {
        
        /** 分類ラベル */
        final int lavel;
        
        /** 特徴量 */
        final double[] feature;
        
        /** スケーリングした特徴量 */
        double[] scalingFeature;
        
        /** 未定乗数 */
        double lambda = 1.;
        
        LearningData(int lavel, double[] feature) {
            this.lavel = lavel;
            this.feature = feature;
        }
        
        /**
         * サポートベクター判定
         */
        boolean isSupportVector() {
            return lambda > 0.0001;
        }
    }
}
