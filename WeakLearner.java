import java.util.Arrays;

public class WeakLearner {
    // immutable
    private final int numDim; // number of attributes per input item, k dimensions

    private final int predictDim; // axis/dimension that stump will predict along
    private final double predictValue; // position of hyperplane
    private final int labelLeft; // predicted label of the data points to the "left" of the hyperplane -- towards 0
    private final int labelRight; // to the right of the hyperplane

    private final double totalError; // 1 - (best weighted sums), used to calculate amount of say
    private final double amountSay; // the "weight" of this weak learner's prediction in boosting algorithm

    // train weak learner
    public WeakLearner(double[][] input, double[] weights, int[] labels, int numClasses) {
        // input is n data points * k dimensions
        // weights for n points, weights are normalized -- between 0 and 1
        // labels for n points

        // corner cases
        if (input == null || weights == null || labels == null)
            throw new IllegalArgumentException("null constructor args");
        int n = input.length;
        int k = input[0].length;
        numDim = k;
        // if rows don't all have same length
        for (double[] array : input)
            if (array.length != k)
                throw new IllegalArgumentException("inconsistent input dimensions");

        // if weights or labels doesn't have same length as input
        if (n != weights.length)
            throw new IllegalArgumentException("missing weight data");
        if (n != labels.length)
            throw new IllegalArgumentException("missing labels data");

        // check for invalid weights and labels
        for (double w : weights)
            if (w < 0)
                throw new IllegalArgumentException("weights must be at least 0");
        for (int label : labels)
            if (label < 0 || label >= numClasses )
                throw new IllegalArgumentException("invalid label");


        // find the combination of d, v, s that maximizes bestLeftWeight and bestRightWeight
        int bestD = -1;
        double bestV = Double.NEGATIVE_INFINITY;
        int bestLeft = -1; // towards 0
        int bestRight = -1;
        double bestLeftWeight = Double.NEGATIVE_INFINITY;
        double bestRightWeight = Double.NEGATIVE_INFINITY;

        for (int d = 0; d < k; d++) { // iterates through k dimensions to find best

            IndexIntPair[] pairArray = new IndexIntPair[n];
            for (int i = 0; i < n; i++) {
                // put in value along dimension
                pairArray[i] = new IndexIntPair(i, input[i][d]);
            }
            Arrays.sort(pairArray); // mergesort O(n log(n)) time, sorts by increasing values

            // each index corresponds to label, contains the weighted sums for each label
            double[] leftWeightedSums = new double[numClasses];
            double[] rightWeightedSums = new double[numClasses];

            // first divider at index 0
            for (int i = 0; i < n; i++) {
                // if same value along dimension as first data point, item is also to the left of dividing line
                if (pairArray[i].value == pairArray[0].value) {
                    // add to the label of the datapoint
                    leftWeightedSums[labels[pairArray[i].index]] += weights[pairArray[i].index];
                }
                else { // to the right of the hyperplane
                    rightWeightedSums[labels[pairArray[i].index]] += weights[pairArray[i].index];
                }

            }

            // update best
            double[] weightInfo = getWeightedMajorities(leftWeightedSums, rightWeightedSums);
            if ( bestD == -1 || (weightInfo[1]+weightInfo[3]) > (bestLeftWeight + bestRightWeight) ){
                bestD = d;
                bestV = pairArray[0].value;
                bestLeft = (int)weightInfo[0];
                bestLeftWeight = weightInfo[1];
                bestRight = (int)weightInfo[2];
                bestRightWeight = weightInfo[3];
            }

            // get weighted sums when hyperplane at other indexes in pairArray -- based on previous weighted sums
            int index = 1;
            while (index < n) {
                // if value along dimension axis is same as previous, sums should not change, so set to same

                // skip to the next index that is a different value
                if (pairArray[index].value == pairArray[index - 1].value) {
                    index++;
                    // no need to update best because weighted sums should be equal to the sums of the previous index
                }
                else {
                    int anchorIndex = index; // tracks where index was when started
                    // continue adding until axis value is different
                    while (index < n && pairArray[index].value == pairArray[anchorIndex].value) {
                        // add to left, subtract from right
                        leftWeightedSums[ labels[pairArray[index].index] ] += weights[pairArray[index].index];
                        rightWeightedSums[ labels[pairArray[index].index] ] -= weights[pairArray[index].index];
                        index++;
                    }

                    // update best
                    weightInfo = getWeightedMajorities(leftWeightedSums, rightWeightedSums);
                    if ( (weightInfo[1]+weightInfo[3]) > (bestLeftWeight + bestRightWeight) ){
                        bestD = d;
                        bestV = pairArray[anchorIndex].value;
                        bestLeft = (int)weightInfo[0];
                        bestLeftWeight = weightInfo[1];
                        bestRight = (int)weightInfo[2];
                        bestRightWeight = weightInfo[3];
                    }

                }
            }

        }

        // set values
        predictDim = bestD;
        predictValue = bestV;
        labelLeft = bestLeft;
        labelRight = bestRight;

        totalError = 1 - (bestLeftWeight + bestRightWeight);

        double useError = totalError;
        if (totalError == 0) // errors of 0 and 1 will make expression diverge so they need to be shifted slightly
            useError = Double.MIN_VALUE; // smallest positive nonzero double value
        else if (totalError == 1)
            useError -= Double.MIN_VALUE;
        amountSay = Math.log10( (1-useError)/ useError) + Math.log10(numClasses - 1) ;

    }

    // using custom simple data structure, stores int value along with an index to more easily work with
    private static class IndexIntPair implements Comparable<IndexIntPair> {
        // store only these variables because weight and label are accessible through other arrays using the index
        private final int index; // the original index the value appears at
        private final double value; // the value

        // set instance variables for the object
        public IndexIntPair(int i, double val) {
            index = i;
            value = val;
        }

        // compare by values, used to sort in value order
        public int compareTo(IndexIntPair other) {
            if (this.value - other.value < 0)
                return -1;
            else if (this.value - other.value > 0)
                return 1;
            return 0;
        }
    }

    // helper method to get the weighted majority class for one side of the hyperplane
    private double[] getWeightedMajorities(double[] left, double[] right){
        int bL = 0;
        int bR = 0; int prevBR = 0;
        double bLVal = 0;
        double bRVal = 0; double prevBRVal = 0;

        for (int i = 0; i<left.length; i++){
            if(left[i] > bLVal){
                bLVal = left[i];
                bL = i;
            }
        }
        for (int i = 0; i<right.length; i++){
            if(right[i] > bRVal){
                prevBRVal = bRVal;
                prevBR = bR;
                bRVal = right[i];
                bR = i;
            }
        }
        if (bL == bR){ // greedy algorithm is sufficient for getting labels
            bR = prevBR;
            bRVal = prevBRVal;
        }

        return new double[]{bL, bLVal, bR, bRVal};
    }

    // return the prediction of the learner for a sample data point
    public int predict(double[] sample) {
        if (sample == null)
            throw new IllegalArgumentException("null sample input");
        if (sample.length != numDim)
            throw new IllegalArgumentException("incompatible input length");

        if (sample[predictDim] <= predictValue)
            return labelLeft; // towards 0
        else
            return labelRight;
    }

    // return dimension the learner uses to separate data
    public int dimPredictor() {
        return predictDim;
    }

    // return value the learner uses to separate data
    public double valuePredictor() {
        return predictValue;
    }

    // return label to the left of the hyperplane the learner uses
    public int leftPredictor() {
        return labelLeft;
    }
    // return label to the right of the hyperplane the learner uses
    public int rightPredictor() {
        return labelRight;
    }

    // return the total error of this weak learner
    public double totalError() {
        return totalError;
    }

    // return the amount of say for this weak learner, used for boosting
    public double amountSay(){
        return amountSay;
    }
}
