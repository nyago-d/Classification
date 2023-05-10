package fxpractice;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

/**
 * さぽーとべくたーましーん
 */
public class SupportVectorMachine implements LearningMachine {
    
    /** 認識したパターン */
    protected final List<LearningData> learning = new ArrayList<>();
    
    /** 学習係数 */
    private final double learningRate = 0.2;
    
    /** マージンの大きさとペナルティのトレードオフ（C>0） */
    protected static final double C = 100000;
    
    /** 最大更新回数 */
    protected final double maxIteration = 10000;
    
    /** 重みベクトル（配列のサイズは入力ベクトルの次元） */
    protected double[] weight;
    
    /** バイアス */
    private double bias;
    
    /** サポートベクター */
    private List<LearningData> supportVectors = null;
    
    /**
     * コンストラクタ
     */
    public SupportVectorMachine(int futureSize) {
        this.weight = new double[futureSize];
    }
    
    /**
     * 教師データを追加
     */
    @Override
    public void add(int cls, double[] data) {
        this.learning.add(new LearningData(cls, data));
    }
    
    /** 
     * 学習 
     */
    @Override
    public void learn() {
        
        // 特徴量スケーリングする
        this.learning.forEach(ld -> ld.scalingFeature = this.scaling(ld.feature));
        
        // 未定乗数を計算
        this.caluculateLambda();
        
        // 未定乗数を確認
        System.out.println(this.learning.stream().map(ld -> String.valueOf(ld.lambda)).collect(Collectors.joining(" ")));

        // サポートベクトルを抜き出し
        this.supportVectors = this.learning.stream().filter(ld -> ld.isSupportVector()).collect(Collectors.toList());

        // 重みベクトルを更新
        this.updateWeightVector();
        
        // バイアスを更新（b = (1 / N_s)・∑(t_n - ∑λ_m・t_m・k(xn・xm)）
        this.updateBias();
    }
    
    /**
     * 未定乗数を計算
     */
    protected void caluculateLambda() {
        
        // 正負のデータに分ける
        Map<Boolean, List<LearningData>> partition = this.learning.stream()
                                                                  .collect(Collectors.partitioningBy(ld -> ld.lavel >= 0));
        List<LearningData> positive = partition.get(true);
        List<LearningData> negative = partition.get(false);
        
        // 勾配法で未定乗数を求める
        for (int i = 0; i < maxIteration; i++) {
            this.gradient(positive, negative);
        }
    }
    
    /**
     * 勾配降下
     */
    private void gradient(List<LearningData> positive, List<LearningData> negative) {
        
        List<LearningData> all = new ArrayList<>();
        all.addAll(positive);
        all.addAll(negative);
        
        // ラムダを更新
        all.stream().forEach(ld -> {
            double delta = this.learningRate * this.lagrange_d(ld, all);
            ld.lambda += delta;
            ld.lambda = this.limit(ld.lambda, 0 , C);
        });
        
        // sum(λ*label)を計算
        double sum = all.stream().mapToDouble(ld -> ld.lambda * ld.lavel).sum();
        
        // KKT条件を満たすために、どこかにしわ寄せする（限りなく0に近くはなるが、近似値なのは仕方ないということで）
        LearningData any = ((sum > 0) ? negative : positive).stream().filter(ld -> ld.isSupportVector()).findAny().get();
        any.lambda -= sum / any.lavel;
    }
    
    /**
     * 値を範囲内に丸める
     */
    private double limit(double val, double min, double max) {
        return Math.max(min, Math.min(val, max));
    }
    
    /**
     * ラグランジュ関数の微分
     * 
     * ∂L/∂a_n = 1 - ∑(λ・tn・tm・k(xn・xm))
     */
    private double lagrange_d(LearningData learningData, List<LearningData> all) {
        
        double sum = all.stream().mapToDouble(ld -> ld.lambda * learningData.lavel * ld.lavel * this.kernel(learningData.scalingFeature, ld.scalingFeature))
                                 .sum();
        
        return 1 - sum;
    }
    
    /**
     * 重みベクトルを更新
     */
    private void updateWeightVector() {
        
        for (int i = 0; i < this.weight.length; i++) {
            final int idx = i;
            this.weight[i] = this.supportVectors.stream()
                                                .mapToDouble(ld -> ld.lambda * ld.lavel * ld.scalingFeature[idx])
                                                .sum();
        }
    }
    
    /**
     * バイアスを更新（b = (1 / N_s)・∑(t_n - ∑λ_m・t_m・k(xn・xm)）
     */
    private void updateBias() {
        
        double sum = 0.;
        for (int n = 0; n < this.supportVectors.size(); n++) {
            LearningData nn = this.supportVectors.get(n);
            double tmp = 0.;
            for (int m = 0; m < this.supportVectors.size(); m++) {
                LearningData mm = this.supportVectors.get(m);
                tmp += mm.lambda * mm.lavel * this.kernel(nn.scalingFeature, mm.scalingFeature);
            }
            sum += nn.lavel - tmp;
        }
        this.bias = sum / this.supportVectors.size();
    }
    
    /**
     * カーネル
     */
    private double kernel(double[] x, double[] y) {
//        return this.dot(x, y);  // 線形カーネル
        return this.gauss(x, y, 5.0);  // ガウスカーネル
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
     * ガウスカーネル
     */
    private double gauss(double[] x, double[] y, double sigma) {
        
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            sum += Math.pow(x[i] - y[i], 2);
        }
        
        return Math.exp(-sum / 2.0 * Math.pow(sigma, 2));
    }
    
    /**
     * 入力データをスケーリングする
     * 
     * <pre>
     * 相変わらずの手抜きスケーリング。
     * </pre>
     */
    private double[] scaling(double[] data) {
        double[] res = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            res[i] = data[i] / 200 - 1;
        }
        return res;
    }

    /**
     * 評価
     */
    @Override
    public int predict(double[] data) {
        return this.sign(this.decision(this.scaling(data)));
    }
    
    private double decision(double[] data) {
        List<LearningData> ldList = this.supportVectors == null ? this.learning : this.supportVectors;
        double sum = ldList.stream().mapToDouble(ld -> ld.lambda * ld.lavel * this.kernel(ld.scalingFeature, data)).sum();
        return sum + this.bias;
    }
    
    private int sign(double in) {
        return in > 0 ? 1 : -1;
    }

    /**
     * 描画する
     */
    @Override
    public void draw(GraphicsContext gc) {
        
        // 全部消す
        gc.clearRect(0, 0, 400, 400);
        
        // 枠だけつくる
        gc.setFill(Color.WHITE);
        gc.setStroke(Color.GREEN);
        gc.fillRect(0, 0, 400, 400);
        gc.strokeRect(0, 0, 400, 400);
        
        // 画面に描画
        this.learning.stream().forEach(ld -> {
            gc.setFill(ld.lavel > 0 ? Color.BLUE : Color.RED);
            gc.fillOval(ld.feature[0], ld.feature[1], 5, 5);
        });
        
        // サポートベクターを丸で囲う
        gc.setStroke(Color.GREEN);
        this.learning.stream().filter(ld -> ld.isSupportVector())
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

    /**
     * この学習機をリセット
     */
    @Override
    public void reset() {
        this.learning.clear();
        this.weight = new double[this.weight.length];
    }
    
    /**
     * タイトル
     */
    @Override
    public String getTitle() {
        return "SVM(SGD)";
    }
    
    /**
     * 学習データ
     */
    protected static class LearningData {
        
        /** 分類ラベル */
        final int lavel;
        
        /** 特徴量 */
        final double[] feature;
        
        /** スケーリングした特徴量 */
        double[] scalingFeature;
        
        /** 未定乗数 */
        double lambda = 1.0;
        
        /**
         * コンストラクタ
         */
        LearningData(int lavel, double[] feature) {
            this.lavel = lavel;
            this.feature = feature;
        }
        
        /**
         * サポートベクター判定
         * 
         * <pre>
         * 確定したわけではなく、候補として残っているという判定であることに注意。
         * 最後まで残っていれば晴れてサポートベクターとして生きていくことになる。
         * </pre>
         */
        boolean isSupportVector() {
            return 0. < lambda && lambda < C;
        }
    }
}
