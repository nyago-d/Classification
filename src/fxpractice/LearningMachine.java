package fxpractice;

import javafx.scene.canvas.GraphicsContext;

public interface LearningMachine {
    
    /** 教師データを追加 */
    void add(int lavel, double[] feature);
    
    /** 学習 */
    void learn();
    
    /** 判定 */
    int predict(double[] data);
    
    /** 描画 */
    void draw(GraphicsContext gc);
    
    /** リセット */
    void reset();
    
    /** タイトル（学習機の名前） */
    String getTitle();
}
