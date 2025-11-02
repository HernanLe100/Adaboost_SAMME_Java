import java.util.ArrayList;
import java.util.Arrays;

public class BoostingAlgorithm {
    // mutable
    private ArrayList<WeakLearner> experts; // stores WeakLearner to be iterated through
    private double[] weights; // current weight of point at index i, changes as algorithm runs
    private final double[][] trainingData; // input data, n*k
    private final int[] dataLabels; // store labels of input
    private final int numClasses; // number of different classifications there are

    public BoostingAlgorithm(double[][] input, int[] labels, int classes) {
        // corner cases
        if (input == null || labels == null)
            throw new IllegalArgumentException("null constructor args");
        int n = input.length; // amount of training data
        int k = input[0].length; // number of dimensions

        // if rows don't all have same length
        for (double[] array : input)
            if (array == null || array.length != k)
                throw new IllegalArgumentException("inconsistent input dimensions");
        // if input doesn't have same length as labels
        if (n != labels.length)
            throw new IllegalArgumentException("missing labels data");
        // check for invalid labels
        numClasses = classes;
        for (int label : labels)
            if (label < 0 && label >= numClasses)
                throw new IllegalArgumentException("invalid label");

        trainingData = input;
        weights = new double[n];
        for (int i = 0; i < n; i++) // set starting weights for each data point, O(n) time
            weights[i] = 1.0 / n;

        dataLabels = Arrays.copyOf(labels, n); // deep copy the labels, O(n)

        experts = new ArrayList<WeakLearner>();
    }

    // return current weight of the ith data point
    public double weightOf(int i) {
        if (i < 0 || i >= weights.length)
            throw new IllegalArgumentException("argument index out of bounds");
        return weights[i];
    }

    // apply one step of boosting algorithm
    public void iterate() {
        // create weak learner using current weights
        WeakLearner wl = new WeakLearner(trainingData, weights, dataLabels, numClasses);
        experts.add(wl);

        // loop takes O(n) time
        for (int i = 0; i < trainingData.length; i++) { // for each data point,
            // incorrectly labeled data get weight increased so next stump tries to include it correctly
            if (wl.predict(trainingData[i]) != dataLabels[i])
                weights[i] = weights[i] * Math.exp( wl.amountSay() );
            // no change to correct weights
        }
        normalizeWeights(); // O(n) time
        // out.println(Arrays.toString(weights));
    }

    // helper method to normalize weights, prevent values from overflowing
    private void normalizeWeights() {
        double sum = 0;
        for (double d : weights)
            sum += d;
        for (int i = 0; i < weights.length; i++)
            weights[i] /= sum;
    }

    // return prediction of the boosting algorithm
    public int predict(double[] sample) {
        if (sample == null)
            throw new IllegalArgumentException("null sample input");
        if (sample.length != trainingData[0].length)
            throw new IllegalArgumentException("incompatible input length");

        // summed weight of experts predicting each label
        double[] sumWeightPredictions = new double[numClasses];

        for (WeakLearner wl : experts) {
            // predict() takes constant time
            sumWeightPredictions[wl.predict(sample)] += wl.amountSay();
        }
        int bestLabel = -1;
        double best = Double.NEGATIVE_INFINITY;
        for(int i =0; i<numClasses; i++){
            if(sumWeightPredictions[i] > best){
                best = sumWeightPredictions[i];
                bestLabel = i;
            }
        }
        return bestLabel;
    }
}
