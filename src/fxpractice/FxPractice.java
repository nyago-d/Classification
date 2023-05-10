package fxpractice;

import java.io.IOException;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

/**
 *
 */
public class FxPractice extends Application {
    
    public static void main(String[] args) {
        launch(args);
    }
    
    @Override
    public void start(Stage primaryStage) throws IOException {
        
        Parent pane = FXMLLoader.load(getClass().getResource("practice.fxml"));
        
        Scene scene = new Scene(pane, 500, 520);
               
        primaryStage.setTitle("機械学習てすと！");
        primaryStage.setScene(scene);
        
        primaryStage.show();
    }
}
