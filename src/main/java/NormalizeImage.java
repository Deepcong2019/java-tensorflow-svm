import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class NormalizeImage {


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
}