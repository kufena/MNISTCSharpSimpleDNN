using System;
using System.Collections.Generic;
using System.Text;

namespace DNN.Activations
{
    public class NullActivation : ActivationFunction
    {
        public double activate(double z)
        {
            return z;
        }

        /**
         * I'm not sure this is actually the derivative of f(z) = z - that should
         * be 1 I guess, but I think this is actually a null activation function so it
         * probably should pass back z'.
         */
        public double derivative(double zprime)
        {
            return zprime;
        }
    }
}
