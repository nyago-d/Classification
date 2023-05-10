package fxpractice;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.util.Pair;

/**
 * 多層パーセプトロン
 * 
 * 入力層 - 隠れ層 - 出力層の3層構造（むしろこれって2層なんじゃ？）
 * 誤差関数Eは二乗誤差とする。(E=1/2sum((y(k) - t(k))^2) 0<=k<ユニット数)
 */
public class MultiLayerPerceotron implements LearningMachine {
    
    /** 認識したパターン */
    private final List<Pair<int[], double[]>> learning = new ArrayList<>();
    
    /** 学習係数 */
    private final double learningRate = 0.2;
    
    /** 入力層の次元 */
    private final int inputDemension;
    
    /** 中間層のユニット数 */
    private final int hiddenCnt;
    
    /** 分類ラベル */
    private final int lavelKind;
    
    /** 隠れ層 */
    private Layer hiddenRayer;
    
    /** 出力層 */
    private Layer outputRayer;
    
    /**
     * コンストラクタ
     * 
     * @param inputDemension    入力の次元
     * @param hiddenCnt         隠し層のユニット数
     * @param lavelKind         分類ラベル
     */
    public MultiLayerPerceotron(int inputDemension, int hiddenCnt, int lavelKind) {
        this.inputDemension = inputDemension;
        this.hiddenCnt = hiddenCnt;
        this.lavelKind = lavelKind;
        this.reset();
    }
    
    /**
     * リセット
     */
    @Override
    public final void reset() {
        this.learning.clear();
        this.hiddenRayer = new Layer(this.hiddenCnt, this.inputDemension, false);
        this.outputRayer = new Layer(this.lavelKind, this.hiddenCnt, true);
    }
    
    /**
     * 教師データを追加
     * 
     * <pre>
     * 正解ラベルは
     * </pre>
     * 
     * @param cls   ラベル
     * @param data  入力値
     */
    @Override
    public void add(int cls, double[] data) {
        int[] label = new int[lavelKind];
        Arrays.fill(label, 0);
        label[cls] = 1; // 正解ラベルのみが発火するのを教師データとする
        this.learning.add(new Pair(label, data));
    }
    
    @Override
    public void learn() {
        
        System.out.println("学習中");
        
        for (int i = 0; i < 5000; i ++) {
            
            for (Pair<int[], double[]> pair : learning) {

                // 入力層は入力＝出力なので、オブジェクト化しない（した方がいい？）
                double[] inputResult = this.addBias(this.scaling(pair.getValue()));

                // 隠し層の計算
                double[] hiddenResult = hiddenRayer.forward(inputResult);
                
                // 出力層の計算
                double[] outputResult = outputRayer.forward(hiddenResult);

                // 出力層での誤差を取得
                double[] outputError = this.getOutputError(pair.getKey(), outputResult);
                
                // 出力層の重みを更新
                this.outputRayer.backward(hiddenResult, outputError);
                
                // 隠し層での誤差を取得
                double[] hiddenError = this.getHiddenError(outputError, hiddenResult);
                
                // 隠し層の重みを更新
                this.hiddenRayer.backward(inputResult, hiddenError);
            }
            
//            for (Pair<int[], double[]> pair : learning) {
//                this.trial(pair.getValue());
//            }
            
//            System.out.println();
        }
        
        System.out.println("完了");
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
        
        double[] result = this.forward(data);
        
        double max = 0.;
        int ans = 0;
        for (int i = 0; i < lavelKind; i++) {
            if (result[i] > max) {
                max = result[i];
                ans = i;
            }
        }
        
        return ans;
    }
    
    private double[] forward(double[] input) {
        
        // 入力層は入力＝出力なので、オブジェクト化しない（した方がいい？）
        double[] inputResult = this.addBias(this.scaling(input));

        // 隠し層の計算
        double[] hiddenResult = hiddenRayer.forward(inputResult);

        // 出力層の計算
        return outputRayer.forward(hiddenResult);
    }

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

    @Override
    public String getTitle() {
         return "多層パーセプトロン";
    }
    
    private double[] addBias(double[] data) {
        double[] d = new double[data.length + 1];
        System.arraycopy(data, 0, d, 0, data.length);
        d[d.length - 1] = 0.5;
        return d;
    }
    
    /**
     * 出力層の誤差を取得
     * 
     * @param classes   教師ラベル
     * @param output    出力層の出力
     * @return 出力層の誤差（次元は出力層のユニット数）
     */
    private double[] getOutputError(int[] classes, double[] output) {
        double[] error = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            error[i] = (output[i] - classes[i]) * this.activation_deriv(output[i]);
//            error[i] = (output[i] - classes[i]);
        }
        return error;
    }
    
    /**
     * 隠し層の誤差を取得
     * 
     * TODO 層を増やすならばレイヤを引数に渡す
     * 
     * @param nextError     次の層の誤差
     * @param hiddenResult  隠し層の出力
     * @return 隠し層の誤差（次元は隠し層のユニット数）
     */
    private double[] getHiddenError(double[] nextError, double[] hiddenResult) {
        
        double[] error = new double[hiddenResult.length];
        for (int i = 0; i < hiddenResult.length; i++) {
            
            Neuron neuron = this.hiddenRayer.neurons.get(i);
            
            // バイアスは重み更新をする必要はない（前の層とつながっていないので）
            if (neuron.isBias()) {
                error[i] = 0;
                
            // 誤差（の出力における偏微分）
            } else {
                
                List<Neuron> neurons = this.outputRayer.neurons;
                
                double sumError = 0.;
                for (int j = 0; j < nextError.length; j++) {
                    sumError += nextError[j] * neurons.get(j).weight[i];
                }
                error[i] = sumError * this.activation_deriv(hiddenResult[i]);
            }
        }
        
        return error;
    }
    
    /**
     * 活性化関数
     */
    private double activation(double x) {
        return this.sigmoid(x);
    }
    
    /**
     * 活性化関数の微分
     */
    private double activation_deriv(double x) {
        return this.sigmoid_deriv(x);
    }
    
    /**
     * シグモイド関数
     */
    private double sigmoid(double x) {
        return 1. / (1. + Math.exp(-x));
    }
    
    /**
     * シグモイド関数の微分
     */
    private double sigmoid_deriv(double x) {
        return x * (1. - x);
    }
    
    /**
     * レイヤ
     */
    private class Layer {
        
        /** このレイヤのニューロン */
        private final List<Neuron> neurons;
        
        /**
         * コンストラクタ
         * 
         * @param size          このレイヤのニューロンの数
         * @param inputSize     前のレイヤからの入力の数
         */
        Layer(int size, int inputSize, boolean isOutput) {
            this.neurons = Stream.generate(() -> new Neuron(inputSize + 1))
                                 .limit(size)
                                 .collect(Collectors.toList());
            if (!isOutput) {
                this.neurons.add(new Neuron(0));
            }
        }
        
        /**
         * 順伝播
         * 
         * @param data  前の層からの入力
         * @return      次の層への出力
         */
        public double[] forward(double[] data) {
            return neurons.stream().mapToDouble(neuron -> neuron.output(data)).toArray();
//            double[] res = new double[data.length - 1];
//            for (int i = 0; i < neurons.size() - 1; i++) {
//                res[i] = neurons.get(i).output(data);
//            }
//            return addBias(res);
        }
        
        /**
         * 逆伝播
         * 
         * @param input この層への入力（前の層の出力）
         * @param error 誤差（の出力における偏微分）
         */
        public void backward(double[] input, double[] errors) {
            for (int i = 0; i < neurons.size(); i++) {
                neurons.get(i).update(input, errors[i]);
            }
        }
        
        @Override
        public String toString() {
            return this.neurons.stream().map(Neuron::toString).collect(Collectors.joining("\n"));
        }
    }
    
    /**
     * ニューロンのモデル
     * 
     * <pre>
     * 入力ベクトルを受け取って重みベクトルを掛け合わせ、
     * 活性化関数を通して出力するのを1つのニューロンの役割だと思っている。
     * クラゲの頭にひもがついた感じ。
     * ---┐
     * ---┼-○---
     * ---┘
     * </pre>
     */
    private class Neuron {
        
        /** 重みベクトル */
        private final double[] weight;
        
        /**
         * コンストラクタ
         * 
         * @param inputSize 入力の次元（前のレイヤの出力数）
         */
        Neuron(int inputSize) {
            this.weight = new double[inputSize];
            for (int i = 0; i < inputSize; i++) {
                this.weight[i] = Math.random()*2-1;
            }
        }
        
        public boolean isBias() {
            return this.weight.length == 0;
        }
        
        /**
         * 出力
         * 
         * <pre>
         * o = f(ωx) = f(u)
         * </pre>
         * 
         * @param input 入力データ
         * @return      出力値
         */
        public double output(double[] input) {
            return MultiLayerPerceotron.this.activation(this.isBias() ? .5 : this.dot(input));
        }
        
        /**
         * 重みベクトルを更新
         * 
         * <pre>
         * ω = ω - ρ・(∂E/∂ω) = ω - ρ・(o(L) - t)・f'(u)・o(L-1)
         * ∂E/∂uを誤差とか呼んでいいの？
         * </pre>
         * 
         * @param input このニューロンへの入力（＝ひとつ前の層の出力）
         * @param error 誤差（の出力における偏微分）
         */
        public void update(double[] input, double error) {
            for (int i = 0; i < this.weight.length; i++) {
                this.weight[i] -= learningRate * error * input[i];
            }
        }
        
        /**
         * 内積の計算
         *
         * @param data  入力データ
         * @return      出力値
         */
        private double dot(double[] data) {
            double res = 0;
            for (int i = 0; i <data.length; i++) {
                res += data[i] * weight[i];
            }
            return res;
        }
        
        @Override
        public String toString() {
            return DoubleStream.of(weight).mapToObj(String::valueOf).collect(Collectors.joining(" "));
        }
    }
}