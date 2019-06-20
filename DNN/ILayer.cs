using MathNet.Numerics.LinearAlgebra;

namespace DNN
{
    public interface ILayer
    {
        ActivationFunction activationFunction { get; set; }
        Vector<double> ayes { get; set; }
        //Vector<double> deriv_ayes { get; set; }

        void activate(Vector<double> prevAyes);
        double L2(Vector<double> expected);
        void resetBiases(IRandomVariable rv);
        void resetWeights(IRandomVariable rv);
        Vector<double> train(Vector<double> upvals, double training_rate);
    }
}