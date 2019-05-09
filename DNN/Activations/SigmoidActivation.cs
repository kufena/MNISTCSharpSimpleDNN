using System;
using System.Collections.Generic;
using System.Text;

namespace DNN.Activations
{
    class SigmoidActivation : ActivationFunction
    {
        public double activate(double z)
        {
            return 1.0 / (1 - Math.Exp(z));
        }

        public double derivative(double zprime)
        {
            double p = activate(zprime);
            return p * (1 - p);
        }
    }
}
