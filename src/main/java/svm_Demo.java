import java.io.IOException;

public class svm_Demo {

    public static void main(String[] args) throws IOException {

        String[] arg = {"-b","1",
                "-s","0",//svm type C-SVC (multi-class classification)
                "-t","0",//liner kernel
                "train.txt", // 训练数据
                "svm_model.txt", // 保存训练模型
                                };

        String[] parg = {"-b","1",
                "test.txt", // 测试数据
                "svm_model.txt", // 调用训练模型
                "predict.txt"}; // 预测结果

        System.out.println("........SVM开始训练..........");
        long start = System.currentTimeMillis();
        svm_train.main(arg); //训练
        System.out.println("训练用时:" + (System.currentTimeMillis() - start));

        System.out.println("........SVM开始预测..........");
        long end = System.currentTimeMillis();
        svm_predict.main(parg); // 预测 //
        System.out.println("测试用时:" + (System.currentTimeMillis() - end));
    }
}

