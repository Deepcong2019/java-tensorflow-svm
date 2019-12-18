# java-tensorflow-svm
invoking tensorflow.pb model and SVM.pmml/txt model by java
## 四种方案（目前实现前两种，基于jpmml调用svm.pmml时只能输出投票分布结果，基于libsvm训练的svm预测才能输出概率分布）：
   * java调用tensorflow pb模型 + java 基于jpmml包 调用在python上训练的SVM.pmml模型
   * java调用tensorflow pb模型 + java调用Java训练的基于libsvm的模型 svm_model.txt
   * java调用tensorflow pmml模型 + Java 基于jpmml包 调用在python上训练的SVM.pmml模型
   * java调用tensorflow pmml模型 + java调用Java训练的基于libsvm的模型 svm_model.txt
   
