package fxpractice;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

public class SVM_SMO implements LearningMachine {
    
    /** 更新閾値 */
    private static final double EPS = 0.001;
    
    /** 認識したパターン */
    protected final List<LearningData> learning = new ArrayList<>();
    
    /** マージンの大きさとペナルティのトレードオフ（C>0） */
    protected static final double C = 1000;
    
    /** 最大更新回数 */
    protected final double maxIteration = 10000;
    
    /** 重みベクトル（配列のサイズは入力ベクトルの次元） */
    protected double[] weight;
    
    /** バイアス */
    private double bias;
    
    /** サポートベクター */
    private List<LearningData> supportVectors = null;

    public SVM_SMO(int futureSize) {
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
        
        // 未定乗数を計算
        this.caluculateLambda();
        
        // ラグランジュ乗数を確認
        System.out.println(this.learning.stream().map(ld -> String.valueOf(ld.lambda)).collect(Collectors.joining(" ")));

        // サポートベクトルを抜き出し
        this.supportVectors = this.learning.stream().filter(ld -> ld.isSupportVector()).collect(Collectors.toList());

        // 重みベクトルを更新
        this.updateWeightVector();
        
        // バイアスを更新
        this.updateBias();
    }
    
    protected void caluculateLambda() {
        
        for (int i = 0; i < maxIteration; i++) {
        
            // 現在のλで計算した値を保持しておく
            this.learning.forEach(ld -> ld.y = this.decision(ld.scalingFeature));
            
            // KKT条件に違反する変数がある
            List<LearningData> alpha2s = this.learning.stream().filter(ld -> ld.checkKkt1())
                                                                .filter(ld -> !ld.checkKkt2())
                                                                .collect(Collectors.toList());
            
            // 0<λ<CでKKT条件を満たさないものがなければ、それ以外でも探す
            if (alpha2s.isEmpty()) {
                alpha2s = this.learning.stream().filter(ld -> !ld.checkKkt2()).collect(Collectors.toList());
            }
            
            // すべてKKT条件を満たしたので終了
            if (alpha2s.isEmpty()) {
                break;
            }
            
            // くるくるする
            boolean isUpdate = true;
            for (LearningData alpha2 : alpha2s) {
                
                // 2つめの変数α1を選択して更新（その1）
                LearningData alpha1_1 = this.findAlpha1_1(alpha2);
                if (this.update(alpha1_1, alpha2)) {
                    break;  // continueしてもいいけど、alpha2sって一部変更かかるよね？
                }
                
                // 2つめの変数α1を選択して更新（その21）
                Optional<LearningData> alpha1_2 = this.findAlpha1_2(alpha2);
                if (alpha1_2.isPresent() && this.update(alpha1_2.get(), alpha2)) {
                    break;
                }
                
                // 2つめの変数α1を選択して更新（その1）
                LearningData alpha1_3 = this.findAlpha1_3(alpha2);
                if (this.update(alpha1_3, alpha2)) {
                    break;
                }
                
                isUpdate = false;
            }
            
            // 何も更新しなくなったら終わりでいい
            if (!isUpdate) {
                break;
            }
        }
    }
    
    /**
     * 2つめの変数α1を探す（その1）
     * 
     * <pre>
     * |E1-E2|を最大化する。
     * これは必ずある。
     * </pre>
     */
    private LearningData findAlpha1_1(LearningData alpha2) {
        return this.learning.stream().filter(ld -> ld != alpha2)  // 自身は除外
                                      .sorted(Comparator.<LearningData>comparingDouble(ld -> Math.abs(ld.e() - alpha2.e())).reversed())
                                      .findFirst().get();
    }
    
    /**
     * 2つめの変数α1を探す（その2）
     * 
     * <pre>
     * 0＜λ＜C
     * これはないかも？
     * </pre>
     */
    private Optional<LearningData> findAlpha1_2(LearningData alpha2) {
        return this.learning.stream().filter(ld -> ld != alpha2)
                                      .filter(ld -> ld.checkKkt1())
                                      .findFirst();
    }
    
    /**
     * 2つめの変数α1を探す（その2）
     * 
     * <pre>
     * ランダム
     * </pre>
     */
    private LearningData findAlpha1_3(LearningData alpha2) {
        LearningData alpha1 = this.learning.get((int) (Math.random() * this.learning.size()));
        return alpha1 != alpha2 ? alpha1 : this.findAlpha1_3(alpha2);
    }
    
    /**
     * λを更新する
     */
    private boolean update(LearningData alpha1, LearningData alpha2) {
        
        // ------ α1の計算 ------
        
        // 上限と下限を先に計算
        double low;
        double high;
        if (alpha1.lavel != alpha2.lavel) {
            low = Math.max(0, alpha1.lambda - alpha2.lambda);
            high = Math.min(C, C - alpha1.lambda + alpha2.lambda);
        } else {
            low = Math.max(0, alpha1.lambda + alpha2.lambda - C);
            high = Math.min(C, alpha1.lambda + alpha2.lambda);
        }
        
        // カーネルを計算
        double k11 = this.kernel(alpha1.scalingFeature, alpha1.scalingFeature);
        double k12 = this.kernel(alpha1.scalingFeature, alpha2.scalingFeature);
        double k22 = this.kernel(alpha2.scalingFeature, alpha2.scalingFeature);
        
        // (k11 - 2 * k12 + k22) <= 0 ⇒ カーネルが負則の場合を考慮すべきなの？
        if ((k11 - 2 * k12 + k22) <= 0) {
            return false;
        }
        
        // α2のλを計算
        double lambda2 = alpha2.lambda + alpha2.lavel * (alpha1.e() - alpha2.e()) / (k11 - 2 * k12 + k22);
        lambda2 = Math.max(low, Math.min(high, lambda2));
        
        // 更新量が小さい場合には更新しない
        if (Math.abs(alpha2.lambda - lambda2) < EPS * (alpha2.lambda + lambda2 + EPS)) { 
            return false; 
        } 

        // α1のλを計算 
        double lambda1 = alpha1.lambda + alpha1.lavel * alpha2.lavel * (alpha2.lambda - lambda2);
        
        // なんか変な値になったら更新しない
        if (Double.isNaN(lambda1) || lambda1 < 0 || C < lambda1) { 
            return false; 
        } 
        
        alpha1.lambda = lambda1;
        alpha2.lambda = lambda2;
        
        return true;
    }
    
       /**
     * 重みベクトルを更新
     */
    private void updateWeightVector() {
        
        for (int i = 0; i < this.weight.length; i++) {
            final int idx = i;
            this.weight[i] = supportVectors.stream()
                                           .mapToDouble(ld -> ld.lambda * ld.lavel * ld.scalingFeature[idx])
                                           .sum();
        }
    }
    
    /**
     * バイアスを更新（b = (1 / N_s)・∑(t_n - ∑λ_m・t_m・k(xn・xm)）
     */
    private void updateBias() {
        
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
     * カーネル
     */
    protected double kernel(double[] x, double[] y) {
//        return this.dot(x, y);  // 線形カーネル
        return this.gauss(x, y, 5.0);  // ガウスカーネル
    }
    
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
//        return this.sign(this.dot(this.scaling(data), weight) + bias);
        return this.sign(this.decision(this.scaling(data)));
    }
    
    protected double decision(double[] data) {
        List<LearningData> ldList = this.supportVectors == null ? this.learning : this.supportVectors;
        double sum = ldList.stream().mapToDouble(ld -> ld.lambda * ld.lavel * this.kernel(ld.scalingFeature, data)).sum();
        return sum + this.bias;
    }
    
    private int sign(double in) {
        return in > 0 ? 1 : -1;
    }

    @Override
    public void draw(GraphicsContext gc) {
        
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
    
    @Override
    public void reset() {
        this.learning.clear();
        this.supportVectors = null;
        this.weight = new double[this.weight.length];
        this.bias = 0.;
    }
    
    @Override
    public String getTitle() {
        return "SVM";
    }
    
    /**
     * 学習データ
     */
    protected static class LearningData {
        
        /** 分類ラベル */
        protected final int lavel;
        
        /** 特徴量 */
        protected final double[] feature;
        
        /** スケーリングした特徴量 */
        protected double[] scalingFeature;
        
        /** 未定乗数 */
        protected double lambda = 0.;
        
        /** 現在の未定乗数で計算した決定関数 */
        protected double y = 0.;
        
        protected LearningData(int lavel, double[] feature) {
            this.lavel = lavel;
            this.feature = feature;
        }
        
        /**
         * KKT条件チェック1
         * 
         * <pre>
         * 0＜λ＜C
         *</pre>
         */
        protected boolean checkKkt1() {
            return 0 < this.lambda && this.lambda < C;
        }
        
        /**
         * KKT条件チェック2
         * 
         * <pre>
         * λ = 0   ⇒ ty≧1
         * 0＜λ＜C ⇒ ty = 1
         * λ = C   ⇒ ty ≦ 1
        * </pre>
        */
        protected boolean checkKkt2() {

            // λ = 0 のとき
            if (this.lambda == 0) {
                return this.lavel * this.y >= 1.; 

            // 0 < λ < C のとき
            } else if (0 < this.lambda && this.lambda < C) { 
                return this.lavel * this.y == 1.; 

            // λ = C のとき
            } else if (this.lambda == C) { 
                return this.lavel * this.y <= 1.; 
            } 

            // どれでもないのは 0 < λ < C のKTT条件を満たしてないからいいのかな？
            return false; 
        }
        
        /**
         * e = y - t
         */
        protected double e() {
            return this.y - this.lavel;
        }
        
        /**
         * サポートベクター判定
         */
        protected boolean isSupportVector() {
            return 0. < lambda && lambda < C;
        }
    }
}