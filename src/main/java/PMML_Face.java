import org.apache.commons.io.IOUtils;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import java.nio.file.Paths;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;
import org.tensorflow.Output;
import org.jpmml.evaluator.*;
import java.util.LinkedHashMap;
import java.io.File;
import java.io.InputStreamReader;
import java.io.BufferedReader;
import java.io.*;
import java.util.ArrayList;
import org.dmg.pmml.FieldName;

public class PMML_Face {

//读取txt文件存为列表
    public static List<String> getFileContent(String path) {
        List<String> strList = new ArrayList<String>();
        File file = new File(path);
        InputStreamReader read = null;
        BufferedReader reader = null;
        try {
            read = new InputStreamReader(new FileInputStream(file),"utf-8");
            reader = new BufferedReader(read);
            String line;
            while ((line = reader.readLine()) != null) {
                strList.add(line);
            }
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (read != null) {
                try {
                    read.close();
                } catch (IOException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }

        }
        return strList;
    }

//预处理图像
    public static Tensor<Float> normalizeImage(final byte[] imageBytes) {
        try (Graph graph = new Graph()) {
            GraphBuilder graphBuilder = new GraphBuilder(graph);

            final Output<Float> output =
                    graphBuilder.div( // Divide each pixels with the MEAN
                            graphBuilder.sub(
                                    graphBuilder.resizeBilinear( // Resize using bilinear interpolation
                                            graphBuilder.expandDims( // Increase the output tensors dimension
                                                    graphBuilder.cast( // Cast the output to Float
                                                            graphBuilder.decodeJpeg(
                                                                    graphBuilder.constant("input", imageBytes), 3),
                                                            Float.class),
                                                    graphBuilder.constant("make_batch", 0)),
                                            graphBuilder.constant("size", new int[]{160, 160})),
                                    graphBuilder.constant("differ", 127.5f)),
                            graphBuilder.constant("scale", 128f));

            try (Session session = new Session(graph)) {
                return session.runner().fetch(output.op().name()).run().get(0).expect(Float.class);
            }
        }
    }

//    // 使用模型生成概率分布
//    private static ProbabilityDistribution getProbabilityDistribution(Evaluator evaluator,Map<FieldName, ?> arguments) {
//        Map<FieldName, ?> evaluateResult = evaluator.evaluate(arguments);
//        List<TargetField> targetFields = evaluator.getTargetFields();
//
//        TargetField targetField = targetFields.get(0);
//
//        FieldName targetFieldName = targetField.getName();
//
//        Object targetFieldValue = evaluateResult.get(targetFieldName);
//
//        return (ProbabilityDistribution) targetFieldValue;
//    }
//
//    // 预测不同分类的概率
//    public static ValueMap<String, Number> predictProba(Evaluator evaluator,Map<FieldName, ?> arguments) {
//        ProbabilityDistribution probabilityDistribution = getProbabilityDistribution(evaluator, arguments);
//        return probabilityDistribution.getValues();
//    }
//
//    // 预测结果分类
//    public Object predict(Evaluator evaluator,Map<FieldName, ?> arguments) {
//        ProbabilityDistribution probabilityDistribution = getProbabilityDistribution(evaluator,arguments);
//        return probabilityDistribution.getPrediction();
//    }

    public static void main(String[] args) throws IOException {
        //加载图片
        String image_path = "cbw.jpg";
        byte[] image = Files.readAllBytes(Paths.get(image_path));
        //图片预处理
        Tensor<Float> normalize_image = normalizeImage(image);
        System.out.println("normalize_image: " + normalize_image);
        //提取特征向量
        try (Graph graph = new Graph()) {
            //导入图
            byte[] graphBytes = IOUtils.toByteArray(new FileInputStream("20180402-114759.pb"));
            graph.importGraphDef(graphBytes);
            //根据图建立Session
            try (Session session = new Session(graph)) {
                Tensor embedding = session.runner()
                        .feed("input:0", normalize_image)
                        .feed("phase_train:0", Tensor.create(false))
                        .fetch("embeddings:0").run().get(0);
                System.out.println("embeddings： " + embedding);

                //特征向量转为数组
                final long[] rshape = embedding.shape();
                float[][] array = new float[1][512];
                embedding.copyTo(array);
                System.out.println("array_length： " +array[0].length);
                for (int i = 0; i<array[0].length;i++){
                    System.out.println("array:" +array[0][i]);
                }

                //制作pmml所需map
                String str=null;
                Map map = new LinkedHashMap();
                for (int i = 0; i<array[0].length;i++){
                    str=String.format("x%d",i+1);
                    map.put(str,array[0][i]);
                }
                int size = map.size();
                System.out.println("Map集合的大小为:" + size);
                System.out.println("Map:" + map);



                //调用模型进行预测，输出最大索引
                PMMLDemo demo = new PMMLDemo();
                Evaluator model = demo.loadPmml();


                Integer name_index = demo.predict(model,map);
                System.out.println("name_index:"+name_index);
                //读取预测人员名单
                List<String> fileContent = getFileContent("class_names.txt");
                for (String name : fileContent) {
                    System.out.println(name);
                }

                //input、output对比
                String pre_name = fileContent.get(name_index);
                System.out.println("input_name:"+image_path);
                System.out.println("out_name:"+pre_name);
                

            }
        }

    }
}


