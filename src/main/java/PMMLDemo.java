import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.*;
import org.xml.sax.SAXException;
import javax.xml.bind.JAXBException;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;


/**
 * Created by Deepcong 2019/12/10.
 */
public class PMMLDemo {
    public Evaluator loadPmml() {
        PMML pmml = new PMML();
        InputStream inputStream = null;
        try {
            inputStream = new FileInputStream("svc.pmml");
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (inputStream == null) {
            return null;
        }
        InputStream is = inputStream;
        try {
            pmml = org.jpmml.model.PMMLUtil.unmarshal(is);
        } catch (SAXException e1) {
            e1.printStackTrace();
        } catch (JAXBException e1) {
            e1.printStackTrace();
        } finally {
            //关闭输入流
            try {
                is.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        ModelEvaluatorFactory modelEvaluatorFactory = ModelEvaluatorFactory.newInstance();
        Evaluator evaluator = modelEvaluatorFactory.newModelEvaluator(pmml);

        pmml = null;
        return evaluator;
    }


    public int predict(Evaluator evaluator, Map map) {

        List<InputField> inputFields = evaluator.getInputFields();
        //过模型的原始特征，从画像中获取数据，作为模型输入
        Map<FieldName, FieldValue> arguments = new LinkedHashMap<FieldName, FieldValue>();
        for (InputField inputField : inputFields) {
            FieldName inputFieldName = inputField.getName();
            Object rawValue = map.get(inputFieldName.getValue());
            FieldValue inputFieldValue = inputField.prepare(rawValue);
            arguments.put(inputFieldName, inputFieldValue);
        }
        System.out.println("arguments: " + arguments);

        Map<FieldName, ?> results = evaluator.evaluate(arguments);
        System.out.println("results: " + results);


        List<TargetField> targetFields = evaluator.getTargetFields();
        System.out.println("targetFileds: " + targetFields);

        TargetField targetField = targetFields.get(0);
        System.out.println("targetFiled: " + targetField);

        FieldName targetFieldName = targetField.getName();
        System.out.println("targetFiledName: " + targetFieldName);

        Object targetFieldValue = results.get(targetFieldName);
        System.out.println("targetFileValue: " + targetFieldValue);

        System.out.println("target: " + targetFieldName.getValue() + " value: " + targetFieldValue);
        int primitiveValue = -1;
        //　　instanceof 严格来说是Java中的一个双目运算符，用来测试一个对象是否为一个可计算的实例，
        if (targetFieldValue instanceof Computable) {
            Computable computable = (Computable) targetFieldValue;
            System.out.println("computable: " + computable);

            primitiveValue = (Integer) computable.getResult();
        }

        return primitiveValue;
    }
}