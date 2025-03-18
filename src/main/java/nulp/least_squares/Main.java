package nulp.least_squares;

public class Main {
    public static void main(String[] args) {
        // Вхідні дані
        double[][] X = {
                {1, 0, 1.5}, {1, 0, 2.5}, {1, 0, 3.5},
                {1, 1, 1.5}, {1, 1, 3.5}, {1, 2, 1.5},
                {1, 2, 2.5}, {1, 2, 3.5}
        };
        double[] y = {2.3, 9.4, 0.2, 1.4, 0.4, 9.7, 4.7, 7.2};

        // Обчислення (X^T * X)
        double[][] XT = transpose(X);
        double[][] XTX = multiplyMatrices(XT, X);

        // Обчислення (X^T * y)
        double[] XTy = multiplyMatrixVector(XT, y);

        // Обчислення коефіцієнтів методом найменших квадратів
        double[][] XTX_inv = invertMatrix(XTX);
        double[] theta = multiplyMatrixVector(XTX_inv, XTy);

        // Обчислення прогнозованих значень
        double[] y_pred = multiplyMatrixVector(X, theta);

        // Обчислення коефіцієнта детермінації R^2
        double r2 = computeR2(y, y_pred);

        // Виведення коефіцієнтів
        System.out.printf("Знайдені коефіцієнти: a0 = %.3f, a1 = %.3f, a2 = %.3f%n",
                theta[0], theta[1], theta[2]);

        // Прогноз для точки x1 = 1.5, x2 = 3
        double[] xTest = {1, 1.5, 3};
        double yPred = dotProduct(theta, xTest);
        System.out.printf("Прогнозоване значення y для x1 = 1.5, x2 = 3: %.3f%n", yPred);

        // Виведення коефіцієнта детермінації
        System.out.printf("Коефіцієнт детермінації R^2: %.3f%n", r2);
    }

    // Транспонування матриці
    public static double[][] transpose(double[][] matrix) {
        int rows = matrix.length, cols = matrix[0].length;
        double[][] transposed = new double[cols][rows];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                transposed[j][i] = matrix[i][j];
        return transposed;
    }

    // Множення матриць
    public static double[][] multiplyMatrices(double[][] A, double[][] B) {
        int rows = A.length, cols = B[0].length, common = B.length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                for (int k = 0; k < common; k++)
                    result[i][j] += A[i][k] * B[k][j];
        return result;
    }

    // Множення матриці на вектор
    public static double[] multiplyMatrixVector(double[][] matrix, double[] vector) {
        int rows = matrix.length, cols = matrix[0].length;
        double[] result = new double[rows];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i] += matrix[i][j] * vector[j];
        return result;
    }

    // Обчислення оберненої матриці методом Гауса-Жордана
    public static double[][] invertMatrix(double[][] matrix) {
        int n = matrix.length;
        double[][] augmented = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(matrix[i], 0, augmented[i], 0, n);
            augmented[i][n + i] = 1;
        }
        for (int i = 0; i < n; i++) {
            double pivot = augmented[i][i];
            for (int j = 0; j < 2 * n; j++) {
                augmented[i][j] /= pivot;
            }

            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = augmented[k][i];
                    for (int j = 0; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }
        double[][] inverse = new double[n][n];
        for (int i = 0; i < n; i++)
            System.arraycopy(augmented[i], n, inverse[i], 0, n);
        return inverse;
    }

    // Обчислення скалярного добутку
    public static double dotProduct(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }

    // Обчислення коефіцієнта детермінації R^2
    public static double computeR2(double[] y, double[] y_pred) {
        double meanY = 0;
        for (double v : y) meanY += v;
        meanY /= y.length;

        double ssTotal = 0, ssResidual = 0;
        for (int i = 0; i < y.length; i++) {
            ssTotal += Math.pow(y[i] - meanY, 2);
            ssResidual += Math.pow(y[i] - y_pred[i], 2);
        }

        return 1 - (ssResidual / ssTotal);
    }
}
