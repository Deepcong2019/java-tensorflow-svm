import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import org.apache.commons.io.IOUtils;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;
import java.io.File;
import java.io.InputStreamReader;
import java.io.BufferedReader;


public class SVM_Face {


    private static double atof(String s) { return Double.valueOf(s).doubleValue(); }

    private static int atoi(String s) { return Integer.parseInt(s); }

    public static void main(String[] args) throws IOException {

        //加载Tensorflow训练模型，提取特征向量：

        //提取特征向量
        Graph graph = new Graph();
        //导入图
        byte[] graphBytes = IOUtils.toByteArray(new FileInputStream("20180402-114759.pb"));
        graph.importGraphDef(graphBytes);
        //根据图建立Session
        Session session = new Session(graph);

        //加载图片
        String image_path = "wuyu.png";
        byte[] image = Files.readAllBytes(Paths.get(image_path));
        //图片预处理
        Tensor<Float> normalize_image = NormalizeImage.normalizeImage(image);
        System.out.println("normalize_image: " + normalize_image);

        //提取特征向量
        Tensor embedding = session.runner().feed("input:0", normalize_image).feed("phase_train:0", Tensor.create(false)).fetch("embeddings:0").run().get(0);
        System.out.println("embeddings： " + embedding);

        //svm对特征向量进行分类，输出识别结果：
        //特征向量转为svm所需格式
        float[][] array = new float[1][512];
        embedding.copyTo(array);
        System.out.println("array_length： " + array[0].length);
        String input = "0";
        String index_value;
        for (int i = 0; i < array[0].length; i++) {
            index_value = String.format("%d:", i);
            input += " " + index_value + array[0][i] + " "; }
        System.out.println("input： " + input);
        //svm分类
        StringTokenizer st = new StringTokenizer(input, " \t\n\r\f:");
        double target = atof(st.nextToken());
        int m = st.countTokens() / 2;
        svm_node[] x = new svm_node[m];
        for (int j = 0; j < m; j++) {
            x[j] = new svm_node();
            x[j].index = atoi(st.nextToken());
            x[j].value = atof(st.nextToken()); }
        svm_model model = svm.svm_load_model("svm_model.txt");
        int nr_class = svm.svm_get_nr_class(model);
        int[] labels = new int[nr_class];
        svm.svm_get_labels(model, labels);

        double[] prob_estimates ;
        prob_estimates = new double[nr_class];
        double v = svm.svm_predict_probability(model, x, prob_estimates);


        List list ;
        list = new ArrayList();
        for (int i = 0; i < prob_estimates.length; i++) { list.add(prob_estimates[i]); }
        System.out.println("probability list：" + list);
        System.out.println("max probability：" + Collections.max(list));


        int name_index = list.indexOf(Collections.max(list));
        System.out.println("max probability index：" + name_index);

        //读取预测人员名单
        List<String> fileContent = new ArrayList();
        File file = new File("class_names.txt");
        InputStreamReader read = new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8);
        BufferedReader reader = new BufferedReader(read);
        String line;
        while ((line = reader.readLine()) != null) { fileContent.add(line); }
        for (String name : fileContent) { System.out.println(name); }

        //input、output对比
        String pre_name = fileContent.get(name_index);
        System.out.println("input_name:" + image_path);
        System.out.println("out_name:" + pre_name);

    }
}


