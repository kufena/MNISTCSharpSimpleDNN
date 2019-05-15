using System;
using System.Collections.Generic;
using System.Text;

namespace DNN.Activations
{
    class SigmoidActivation : ActivationFunction
    {
        public double activate(double z)
        {
            var e_x = Math.Exp(z);
            return e_x / (e_x + 1);
        }

        public double derivative(double zprime)
        {
            double p = activate(zprime);
            return p * (1 - p);
        }
    }
}
