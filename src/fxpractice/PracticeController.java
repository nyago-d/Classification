package fxpractice;

import java.net.URL;
import java.util.ResourceBundle;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
import javafx.scene.control.ToggleGroup;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;

/**
 *
 * @author daiki
 */
public class PracticeController implements Initializable {
    
    /** タイトル */
    @FXML
    private Label title;
    
    /** キャンバス */
    @FXML 
    private Canvas canvas;
    
    /** チェックボックス */
    @FXML
    private ToggleGroup toggle;
    
//    /** プルダウン */
//    @FXML
//    private ComboBox pull;
    
    /** 学習機 */
    private final LearningMachine lm = new SinglePerceptron(2);
    
    /**
     * クリアボタン
     */
    @FXML
    protected void handleClearButton(ActionEvent event) {
        this.clear();
    }
    
    /**
     * 学習ボタン
     */
    @FXML
    protected void handleLearnButton(ActionEvent event) {
        
        GraphicsContext gc = this.canvas.getGraphicsContext2D();
        
        // 学習する
        this.lm.learn();

        // 描画する
        this.lm.draw(gc);
    }
    
//    /**
//     * プルダウン変更
//     */
//    @FXML
//    protected void handlePull(ActionEvent event) {
//        if (this.lm instanceof NearestNeighbor) {
////            NearestNeighbor.class.cast(this.lm).setK(Integer.parseInt(pull.getSelectionModel().getSelectedItem().toString()));
//        }
//    }
    
    /**
     * キャンバスクリック
     */
    @FXML
    protected void handleCanvasClick(MouseEvent event) {
        
        // 赤と青のどちらを選んでいるか（教師ラベル）
        int val = Integer.parseInt((String)toggle.getSelectedToggle().getUserData());
        
        // 教師ラベルと教師データを設定
        this.lm.add(val, new double[]{event.getX(), event.getY()});
        
        // 画面に描画
        GraphicsContext gc = this.canvas.getGraphicsContext2D();
        gc.setFill(val > 0 ? Color.BLUE : Color.RED);
        gc.fillOval(event.getX(), event.getY(), 5, 5);
    }
    
    /**
     * 初期化
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        
        // クリア
        this.clear();
        
        // タイトルを設定
        this.title.setText(lm.getTitle());
    }
    
//    private void doProt(int x, int y, int lavel) {
//    
//        // 教師ラベルと教師データを設定
//        this.lm.add(lavel, new double[]{x, y});
//        
//        // 画面に描画
//        GraphicsContext gc = this.canvas.getGraphicsContext2D();
//        gc.setFill(lavel > 0 ? Color.BLUE : Color.RED);
//        gc.fillOval(x, y, 5, 5);
//    }
    
    /**
     * クリア
     */
    private void clear() {
        
        GraphicsContext gc = canvas.getGraphicsContext2D();
        
        // 全部消す
        gc.clearRect(0, 0, 400, 400);
        
        // 枠だけつくる
        gc.setFill(Color.WHITE);
        gc.setStroke(Color.GREEN);
        gc.fillRect(0, 0, 400, 400);
        gc.strokeRect(0, 0, 400, 400);
        
        // 学習機もリセットする
        this.lm.reset();
    }
}
