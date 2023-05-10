package fxpractice;

import java.util.ArrayList;
import java.util.List;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

public class SinglePerceptron implements LearningMachine {

    /** 認識したパターン */
    private final List<LearningData> learningDataList = new ArrayList<>();
    
    /** 最大更新回数 */
    private int maxIteration = 1000;
    
    /** 重みベクトル（式中の記号だとwと書かれることが多い） */
    private double[] weight;
    
    /** 学習率 */
    private final double learningRate = 0.3;
    
    /**
     * コンストラクタ
     * 
     * @param featureSize 入力（ベクトル）の次元
     */
    public SinglePerceptron(int featureSize) {
        this.weight = new double[featureSize + 1];  // +1はバイアスの重み
    }
    
    /**
     * 最大更新回数を設定
     */
    public SinglePerceptron setMaxIteration(int maxIteration) {
        this.maxIteration = maxIteration;
        return this;
    }
    
    /**
     * 教師データを追加
     */
    @Override
    public void add(int lavel, double[] feature) {
        this.learningDataList.add(new LearningData(lavel, feature));
    }
    
    /** 
     * 学習 
     */
    @Override
    public void learn() {
        
        // 非線形分離の場合、解なしなので上限あり
        for (int j = 0; j < maxIteration; j++) {
            
            boolean change = false;
            for (LearningData ld : this.learningDataList) {
                
                // 入力ベクトルをスケーリングしてバイアスを足す
                double[] input = this.addBias(this.scaling(ld.feature));

                // 出力してみる
                int answer = this.sign(this.dot(input, this.weight));
                
                // 出力とラベルが一致していれば更新する必要はない
                if (answer == ld.lavel) {
                    continue;
                }
                
                // 重みベクトルを更新
                for (int i = 0; i < this.weight.length; i++) {
                    this.weight[i] += this.learningRate * ld.lavel * input[i];
                }
                change = true;
            }
            
            // すべての重みが更新されなくなったら終了
            if (!change) {
                return;
            }
        }
        
        System.out.println("解なし");
    }
    
    /**
     * 評価
     */
    @Override
    public int predict(double[] feature) {
        return this.sign(this.dot(this.addBias(this.scaling(feature)), this.weight));
    }
    
    /**
     * 内積の計算
     */
    private double dot(double[] x, double[] y) {
        double res = 0;
        for (int i = 0; i < x.length; i++) {
            res += x[i] * y[i];
        }
        return res;
    }
    
    /**
     * 活性化関数
     */
    private int sign(double val) {
        return val >= 0 ? 1 : -1;
    }

    /**
     * 描画する
     */
    @Override
    public void draw(GraphicsContext gc) {
        
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
    
    /**
     * 特徴量をスケーリングする
     * 
     * <pre>
     * 手抜きスケーリング。
     * </pre>
     */
    private double[] scaling(double[] feature) {
        double[] res = new double[feature.length];
        for (int i = 0; i < feature.length; i++) {
            res[i] = feature[i] / 200 - 1;
        }
        return res;
    }
    
    /**
     * バイアスを追加
     * 
     * <pre>
     * バイアスは前でも後ろでもいい。
     * </pre>
     */
    private double[] addBias(double[] feature) {
        double[] d = new double[feature.length + 1];
        System.arraycopy(feature, 0, d, 0, feature.length);
        d[d.length - 1] = 1;
        return d;
    }

    /**
     * この学習機をリセット
     */
    @Override
    public void reset() {
        this.learningDataList.clear();
        this.weight = new double[this.weight.length];
    }
    
    /**
     * タイトル
     */
    @Override
    public String getTitle() {
        return "単純パーセプトロン";
    }
    
    /**
     * 学習データ
     */
    private static class LearningData {
        
        /** 分類ラベル（式中の記号だとtと書かれることが多い） */
        final int lavel;
        
        /** 特徴量（式中の記号だとxと書かれることが多い） */
        final double[] feature;
        
        /**
         * コンストラクタ
         * 
         * @param lavel     分類ラベル
         * @param feature   特徴量
         */
        LearningData(int lavel, double[] feature) {
            this.lavel = lavel;
            this.feature = feature;
        }
    }
}
