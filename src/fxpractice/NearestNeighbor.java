package fxpractice;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.util.Pair;

/**
 * k-近傍法
 */
public class NearestNeighbor implements LearningMachine {
    
    /** 最近傍採用数 */
    private final int k;
    
    /** 認識したパターン */
    private final List<Pair<Integer, double[]>> learning = new ArrayList<>();

    /**
     * コンストラクタ
     */
    public NearestNeighbor(int k) {
        this.k = k;
    }
    
    /**
     * 教師データを追加
     */
    @Override
    public void add(int cls, double[] data) {
        this.learning.add(new Pair(cls, data));
    }
    
    /** 
     * 学習 
     */
    @Override
    public void learn() {
    }

    /**
     * 描画
     */
    @Override
    public void draw(GraphicsContext gc) {
        
        int w = (int) gc.getCanvas().getWidth();
        int h = (int) gc.getCanvas().getHeight();
        
        for (int x = 0; x < w; x += 2) {
            for (int y = 0; y < h; y += 2) {
                int ans = this.predict(new double[]{x, y});
                if (ans == 0) {
                    continue;
                }
                gc.setFill(ans > 0 ? Color.BLUE : Color.RED);
                gc.fillOval(x, y, 1, 1);
            }
        }
    }
    
    /**
     * 評価
     */
    @Override
    public int predict(double[] data) {
        
        // 距離とクラスのランキング
        Map<Double, Integer> ranking = new TreeMap<>();
        
        // 一番近いパターンを求める
        this.learning.stream().filter(entry -> entry.getValue().length == data.length)
                              .forEach((entry) -> {
            
            double[] pos = entry.getValue();
            
            // データ間の距離を求める（平方ユークリッド距離）
            double dist = Stream.iterate(0, i -> i + 1).limit(pos.length)
                                                       .mapToDouble(i -> Math.pow(pos[i] - data[i], 2))
                                                       .sum();

            // ソートして貯める（ベクトルは特に必要ないので距離とクラス）
            ranking.put(dist, entry.getKey());
        });
        
        // クラスと個数でマッピング
        Map<Integer, Long> map = ranking.entrySet().stream().limit(k)
                   .collect(Collectors.groupingBy(x -> x.getValue(), Collectors.counting()));
        
        // 最大のクラスを取得
        return map.entrySet().stream().filter(e -> e.getValue().longValue() == Collections.max(map.values()).longValue())
                                      .mapToInt(e -> e.getKey())
                                      .findFirst().orElse(0);
    }
    
    /**
     * リセット
     */
    @Override
    public void reset() {
        this.learning.clear();
    }

    /**
     * タイトル
     */
    @Override
    public String getTitle() {
        return (k == 1 ? "最" : k) + "近傍法";
    }
}